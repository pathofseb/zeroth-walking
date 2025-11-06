"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
import xax
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import Model, ZbotWalkingTask


def rotate_quat_by_quat(
    quat_to_rotate: Array,
    rotating_quat: Array,
    *,
    inverse: bool = False,
    eps: float = 1e-6,
) -> Array:
    """Return rotating_quat * quat_to_rotate * rotating_quat⁻¹ (optionally inverse)."""
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)

    if inverse:
        rotating_quat = jnp.concatenate([rotating_quat[..., :1], -rotating_quat[..., 1:]], axis=-1)

    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = jnp.concatenate([w, x, y, z], axis=-1)
    return out / (jnp.linalg.norm(out, axis=-1, keepdims=True) + eps)


# Command names: vx, vy, wz, heading, bh, rx, ry
COMMAND_NAMES = ["vx", "vy", "wz", "heading", "bh", "rx", "ry"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: ZbotWalkingTask = ZbotWalkingTask.load_task(ckpt_path)

    # Load mujoco model first for init_params
    mujoco_model = task.get_mujoco_model()

    # Create init params for model loading
    init_params = ksim.InitParams(key=jax.random.PRNGKey(0), physics_model=mujoco_model)
    model: Model = task.load_ckpt(ckpt_path, part="model", init_params=init_params)[0]

    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.
    carry_shape = (task.config.depth, task.config.hidden_size)

    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        # Compute projected gravity from quaternion (IMU simulation)
        # Gravity is [0, 0, -1] in world frame, rotate by inverse quaternion
        gravity = jnp.array([0.0, 0.0, -1.0])
        # Quaternion rotation: q^-1 * v * q
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

        dist, carry = model.actor.forward(obs, carry)
        # For MixtureSameFamily, get the mean of the highest-weight component
        # dist.mixture_distribution is Categorical with logits
        # dist.components_distribution is Normal with loc and scale
        mixture_probs = jax.nn.softmax(dist.mixture_distribution.logits, axis=-1)  # shape: (num_joints, num_mixtures)
        best_component_idx = jnp.argmax(mixture_probs, axis=-1)  # shape: (num_joints,)

        # Get means from components - shape: (num_joints, num_mixtures)
        component_means = dist.components_distribution.loc

        # Select the mean from the best component for each joint
        action = jnp.take_along_axis(component_means, best_component_idx[:, None], axis=1).squeeze(-1)

        return action, carry

    metadata = PyModelMetadata(
        joint_names=joint_names,
        command_names=COMMAND_NAMES,
        carry_size=carry_shape,
    )

    # JIT the functions - jax.jit() returns Wrapped objects that kinfer expects
    init_wrapped = jax.jit(init_fn)
    step_wrapped = jax.jit(step_fn)

    init_onnx = export_fn(init_wrapped, metadata)
    step_onnx = export_fn(step_wrapped, metadata)
    kinfer_model = pack(init_onnx, step_onnx, metadata)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(kinfer_model)
    print(f"Kinfer model written to {out_path}")


if __name__ == "__main__":
    main()
