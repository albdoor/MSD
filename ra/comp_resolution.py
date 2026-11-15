from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_LINKS = 5
T_END = 10
DATA_DIR = Path(__file__).resolve().parent / f"data{T_END}s"

FORMULATIONS = {
    "Newton-Euler": "NE",
    "Lagrange": "LE",
    "Featherstone": "Fst",
    "Gibbs-Appell": "Gibbs",
    "Lie Group": "Lie",
}


def load_outputs(n_links, data_dir):
    """Load each formulation's normal and _1000 torque output."""
    outputs = {}
    expected_columns = None

    for formulation, filename_prefix in FORMULATIONS.items():
        normal_path = data_dir / f"torques{filename_prefix}{n_links}.csv"
        high_resolution_path = data_dir / f"torques{filename_prefix}{n_links}_1000.csv"

        if not normal_path.is_file():
            raise FileNotFoundError(f"Normal output not found: {normal_path}")
        if not high_resolution_path.is_file():
            raise FileNotFoundError(f"_1000 output not found: {high_resolution_path}")

        normal = pd.read_csv(normal_path)
        high_resolution = pd.read_csv(high_resolution_path)
        if normal.empty or high_resolution.empty:
            raise ValueError(f"Torque output is empty for {formulation}.")
        if list(normal.columns) != list(high_resolution.columns):
            raise ValueError(f"Joint columns differ between files for {formulation}.")
        if expected_columns is None:
            expected_columns = list(normal.columns)
        elif list(normal.columns) != expected_columns:
            raise ValueError(f"Joint columns differ between formulations at {formulation}.")
        if len(high_resolution) <= len(normal):
            raise ValueError(
                f"The _1000 output for {formulation} must have more samples "
                "than the normal output."
            )

        outputs[formulation] = (normal, high_resolution)

    return outputs


def compare_outputs(outputs, t_end):
    """Plot both resolutions and report errors at the normal sample times."""
    n_joints = len(next(iter(outputs.values()))[0].columns)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        n_joints, 1, figsize=(15, 3.2 * n_joints), squeeze=False
    )

    for formulation_index, (formulation, (normal, high_resolution)) in enumerate(
        outputs.items()
    ):
        color = colors(formulation_index)
        normal_time = np.linspace(0, t_end, len(normal))
        high_resolution_time = np.linspace(0, t_end, len(high_resolution))
        aligned_high_resolution = {}

        for column in normal.columns:
            aligned_high_resolution[column] = np.interp(
                normal_time, high_resolution_time, high_resolution[column].to_numpy()
            )

        for joint_index, column in enumerate(normal.columns):
            axes[joint_index, 0].plot(
                normal_time,
                normal[column],
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{formulation} normal ({len(normal)} samples)",
            )
            axes[joint_index, 0].plot(
                high_resolution_time,
                high_resolution[column],
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{formulation} _1000 ({len(high_resolution)} samples)",
            )

            difference = (
                normal[column].to_numpy() - aligned_high_resolution[column]
            )
            rms = np.sqrt(np.mean(difference**2))
            reference_rms = np.sqrt(np.mean(aligned_high_resolution[column] ** 2))
            relative_rms = rms / reference_rms if reference_rms else np.nan
            print(
                f"{formulation}, {column}: "
                f"RMS difference={rms:.6f}, relative RMS={relative_rms:.6f}"
            )

    for joint_index, (axis_row,) in enumerate(axes):
        axis_row.set_title(
            f"Joint {joint_index + 1} Torque: Normal vs _1000", fontweight="bold"
        )
        axis_row.set_xlabel("Time (s)")
        axis_row.set_ylabel("Torque (Nm)")
        axis_row.grid(True, which="major", alpha=0.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


if __name__ == "__main__":
    torque_outputs = load_outputs(N_LINKS, DATA_DIR)
    compare_outputs(torque_outputs, T_END)
