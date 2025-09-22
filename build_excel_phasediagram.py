import os
import copy 
import PhaseDiagram as PD
import pandas as pd

def data_to_excel(T_list, delta_list, save_dir, general_dir = "phasediagram", filename = None, result = "difference",
                  threshold = None, regularization = None):

    data_grid = PD.load_saved_results(T_list, delta_list, save_dir = "pd_1_size31_noedge", general_dir = "phasediagram")

    grid = PD.get_data_grid_results(data_grid, T_list, delta_list, result = result)
    grid_threshold, grid_threshold_shot_counts = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result = result, threshold = threshold, count_shots=True)
    grid_regularized = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result = result, regularization = regularization)
    grid_regularized_threshold = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result = result, threshold = threshold, regularization = regularization, count_shots=True)

    # grid is a 2D numpy array (shape: len(delta_list) x len(T_list))
    df = pd.DataFrame(grid, index=delta_list, columns=T_list)
    df_thre = pd.DataFrame(grid_threshold, index=delta_list, columns=T_list)
    df_reg = pd.DataFrame(grid_regularized, index=delta_list, columns=T_list)
    df_reg_thre = pd.DataFrame(grid_regularized_threshold, index=delta_list, columns=T_list)

    # Optionally, rename the index and columns for clarity
    df.index.name = 'delta'
    df.columns.name = 'T'
    df_thre.index.name = 'delta'
    df_thre.columns.name = 'T'
    df_reg.index.name = 'delta'
    df_reg.columns.name = 'T'
    df_reg_thre.index.name = 'delta'
    df_reg_thre.columns.name = 'T'

    if filename is None:
        filename = f"{general_dir}/{save_dir}/"
        filename = f"table"
        filename += ".xlsx"

    # Export to Excel
    with pd.ExcelWriter("phase_diagram_tables.xlsx") as writer:
        df.to_excel(writer, sheet_name="difference", startrow=0)
        # Write the second table below the first (add space if you want)
        df_thre.to_excel(writer, sheet_name="difference", startrow=len(df)+6)
        df_reg.to_excel(writer, sheet_name="difference", startrow=2*(len(df)+6))
        rt.to_excel(writer, sheet_name="ratio", startrow=0)
        # Write the second table below the first (add space if you want)
        rt_thre.to_excel(writer, sheet_name="ratio", startrow=len(rt)+6)
        rt_reg.to_excel(writer, sheet_name="ratio", startrow=2*(len(rt)+6))

    df_thre_shot_counts = pd.DataFrame(difference_grid_threshold_shot_counts, index=delta_list, columns=T_list)
    df_thre_shot_counts.index.name = 'delta'
    df_thre_shot_counts.columns.name = 'T'

    rt_thre_shot_counts = pd.DataFrame(ratio_grid_threshold_shot_counts, index=delta_list, columns=T_list)
    rt_thre_shot_counts.index.name = 'delta'
    rt_thre_shot_counts.columns.name = 'T'

    with pd.ExcelWriter("phase_diagram_remaining_shots.xlsx") as writer:
        df_thre_shot_counts.to_excel(writer, sheet_name="difference", startrow=0)
        rt_thre_shot_counts.to_excel(writer, sheet_name="ratio", startrow=0)


def data_to_excel(
    T_list,
    delta_list,
    save_dir,
    general_dir="phasediagram",
    filename=None,
    results="both",          # "difference", "ratio", or "both"
    threshold=None,          # if None, threshold and shot grids are skipped
    regularization=None      # passed to thresholded computation
):
    """
    Build an Excel workbook with one sheet per result type.
    Each sheet contains:
      - Base grid
      - Threshold grid (if threshold provided)
      - Remaining-shots grid (just below threshold grid)
    """

    # Normalize results argument
    if isinstance(results, str):
        if results == "both":
            results_list = ["difference", "ratio"]
        elif results in ("difference", "ratio"):
            results_list = [results]
        else:
            raise ValueError("results must be 'difference', 'ratio', or 'both'")
    else:
        # Iterable provided
        results_list = list(results)

    # Load saved data
    data_grid = PD.load_saved_results(T_list, delta_list, save_dir=save_dir, general_dir=general_dir)

    # Build output path
    out_dir = os.path.join(general_dir, save_dir)
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        suffix = "-".join(results_list)
        filename = f"phase_diagram_tables_{suffix}.xlsx"
    out_path = os.path.join(out_dir, filename)

    with pd.ExcelWriter(out_path) as writer:
        for res in results_list:
            # Base grid (optionally regularized if you want consistency)
            base_grid = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result=res)
            df_base = pd.DataFrame(base_grid, index=delta_list, columns=T_list)
            df_base.index.name = "delta"
            df_base.columns.name = "T"

            start = 0
            # Small labels for sections (optional)
            pd.DataFrame([[f"{res} - base"]]).to_excel(writer, sheet_name=res, startrow=start, startcol=0, header=False, index=False)
            start += 1
            df_base.to_excel(writer, sheet_name=res, startrow=start)
            start += len(df_base) + 3

            # Threshold + shot grids if threshold is provided
            if threshold is not None:
                thr_grid, shot_counts = PD.get_data_grid_results(
                    copy.deepcopy(data_grid),
                    T_list,
                    delta_list,
                    result=res,
                    threshold=threshold,
                    regularization=regularization,
                    count_shots=True
                )
                df_thr = pd.DataFrame(thr_grid, index=delta_list, columns=T_list)
                df_thr.index.name = "delta"
                df_thr.columns.name = "T"

                pd.DataFrame([[f"{res} - threshold (threshold={threshold}, regularization={regularization})"]]).to_excel(
                    writer, sheet_name=res, startrow=start, startcol=0, header=False, index=False
                )
                start += 1
                df_thr.to_excel(writer, sheet_name=res, startrow=start)
                start += len(df_thr) + 2

                if shot_counts is not None:
                    df_shots = pd.DataFrame(shot_counts, index=delta_list, columns=T_list)
                    df_shots.index.name = "delta"
                    df_shots.columns.name = "T"

                    # Place shot grid directly under the threshold grid
                    pd.DataFrame([[f"{res} - remaining shots under threshold"]]).to_excel(
                        writer, sheet_name=res, startrow=start, startcol=0, header=False, index=False
                    )
                    start += 1
                    df_shots.to_excel(writer, sheet_name=res, startrow=start)

    return out_path