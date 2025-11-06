"""Converts a checkpoint to a deployable model using JAX native export.

This bypasses the kinfer → jax2tf → TensorFlow → ONNX pipeline and uses
JAX's native jax.export API to serialize models in StableHLO format.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jax import export

from train import Model, ZbotWalkingTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: ZbotWalkingTask = ZbotWalkingTask.load_task(ckpt_path)

    # Load mujoco model for init_params
    mujoco_model = task.get_mujoco_model()

    # Create init params for model loading
    init_params = ksim.InitParams(key=jax.random.PRNGKey(0), physics_model=mujoco_model)
    model: Model = task.load_ckpt(ckpt_path, part="model", init_params=init_params)[0]

    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes root joint
    carry_shape = (task.config.depth, task.config.hidden_size)

    print(f"Model loaded:")
    print(f"  Joints: {len(joint_names)}")
    print(f"  Carry shape: {carry_shape}")
    print(f"  Hidden size: {task.config.hidden_size}")
    print(f"  Depth: {task.config.depth}")
    print(f"  Mixtures: {task.config.num_mixtures}")

    # Define the inference function
    def inference_fn(
        joint_angles: jax.Array,
        joint_angular_velocities: jax.Array,
        quaternion: jax.Array,
        command: jax.Array,
        carry: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Run inference on the model.

        Args:
            joint_angles: Joint positions (num_joints,)
            joint_angular_velocities: Joint velocities (num_joints,)
            quaternion: Robot orientation as quaternion (4,)
            command: Command vector [vx, vy, wz, heading, bh, rx, ry] (7,)
            carry: RNN hidden state (depth, hidden_size)

        Returns:
            action: Target joint positions (num_joints,)
            next_carry: Updated RNN state (depth, hidden_size)
        """
        # Compute projected gravity from quaternion (IMU simulation)
        gravity = jnp.array([0.0, 0.0, -1.0])
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        quat_inv = jnp.array([w, -x, -y, -z])

        # Rotate gravity vector by inverse quaternion
        t = 2.0 * jnp.cross(quat_inv[1:], gravity)
        projected_gravity = gravity + quat_inv[0] * t + jnp.cross(quat_inv[1:], t)

        # Build observation matching train.py run_actor with use_imu=True
        obs = jnp.concatenate(
            [
                joint_angles,              # NUM_JOINTS
                joint_angular_velocities,  # NUM_JOINTS
                projected_gravity,         # 3 (IMU)
                command[..., :2],          # vx, vy
                command[..., 3:4],         # heading (index 3!)
                command[..., 4:],          # bh, rx, ry
            ],
            axis=-1,
        )

        dist, next_carry = model.actor.forward(obs, carry)

        # For MixtureSameFamily, get the mean of the highest-weight component
        mixture_probs = jax.nn.softmax(dist.mixture_distribution.logits, axis=-1)
        best_component_idx = jnp.argmax(mixture_probs, axis=-1)
        component_means = dist.components_distribution.loc
        action = jnp.take_along_axis(component_means, best_component_idx[:, None], axis=1).squeeze(-1)

        return action, next_carry

    # Create example inputs with proper shapes
    example_joint_angles = jnp.zeros(len(joint_names))
    example_joint_vels = jnp.zeros(len(joint_names))
    example_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    example_command = jnp.zeros(7)
    example_carry = jnp.zeros(carry_shape)

    # Export the function using JAX's native export
    print("\nExporting model using jax.export...")
    exported = export.export(jax.jit(inference_fn))(
        example_joint_angles,
        example_joint_vels,
        example_quat,
        example_command,
        example_carry,
    )

    # Save the exported model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.jax_export"
    with open(model_path, "wb") as f:
        f.write(exported.serialize())

    # Save metadata
    metadata_path = out_dir / "metadata.json"
    import json
    metadata = {
        "joint_names": joint_names,
        "num_joints": len(joint_names),
        "carry_shape": carry_shape,
        "hidden_size": task.config.hidden_size,
        "depth": task.config.depth,
        "num_mixtures": task.config.num_mixtures,
        "command_names": ["vx", "vy", "wz", "heading", "bh", "rx", "ry"],
        "input_shapes": {
            "joint_angles": [len(joint_names)],
            "joint_angular_velocities": [len(joint_names)],
            "quaternion": [4],
            "command": [7],
            "carry": list(carry_shape),
        },
        "output_shapes": {
            "action": [len(joint_names)],
            "next_carry": list(carry_shape),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel exported successfully:")
    print(f"  Model: {model_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"\nTo load and use:")
    print(f"  from jax import export")
    print(f"  with open('{model_path}', 'rb') as f:")
    print(f"      exported = export.deserialize(f.read())")
    print(f"  result = exported.call(...)")


if __name__ == "__main__":
    main()
