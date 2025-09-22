import os
import copy 
import PhaseDiagram as PD
import pandas as pd


def data_to_excel(
    T_list,
    delta_list,
    save_dir,
    general_dir="phasediagram",
    filename=None,
    results="both",          # "difference", "ratio", or "both"
    threshold=None,          # list [diff_thre, ratio_thre]
    regularization=None      # list [diff_reg, ratio_reg]
):
    """
    Build an Excel workbook with one sheet per result type.
    Each sheet contains:
      - Base grid
      - Threshold grid (if threshold provided)
      - Remaining-shots grid (just below threshold grid)
      - Regularized grid (if regularization provided)
      - Regularized + threshold grid (if both provided)
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

    if isinstance(threshold, list) == False:
        threshold = [threshold] * 2
    
    if isinstance(regularization, list) == False:
        regularization = [regularization] * 2


    # Load saved data
    data_grid = PD.load_saved_results(T_list, delta_list, save_dir=save_dir, general_dir=general_dir)

    # Build output path
    out_dir = os.path.join(general_dir, save_dir, "excels")
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        suffix = "-".join(results_list)
        filename = f"phase_diagram_tables_{suffix}" 
        if threshold is not None:
            filename += f"_thr{threshold}"
        if regularization is not None:
            filename += f"_reg{regularization}"
        filename += ".xlsx"
    out_path = os.path.join(out_dir, filename)


    with pd.ExcelWriter(out_path) as writer:
        for idx, res in enumerate(results_list):
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
                thr_grid, shot_counts = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result=res, threshold=threshold[idx], count_shots=True)
                df_thr = pd.DataFrame(thr_grid, index=delta_list, columns=T_list)
                df_thr.index.name = "delta"
                df_thr.columns.name = "T"

                pd.DataFrame([[f"{res} - threshold (threshold={threshold[idx]})"]]).to_excel(
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
                    start += len(df_shots) + 3
                
            if regularization is not None:
                reg_grid = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result = res, regularization = regularization[idx])
                df_reg = pd.DataFrame(reg_grid, index=delta_list, columns=T_list)
                df_reg.index.name = "delta"
                df_reg.columns.name = "T"

                pd.DataFrame([[f"{res} - regularization (regularization={regularization[idx]})"]]).to_excel(
                    writer, sheet_name=res, startrow=start, startcol=0, header=False, index=False)
                start += 1
                df_reg.to_excel(writer, sheet_name=res, startrow=start)
                start += len(df_reg) + 2

            if regularization is not None and threshold is not None:
                thr_reg_grid, shot_counts_reg = PD.get_data_grid_results(copy.deepcopy(data_grid), T_list, delta_list, result = res, threshold = threshold[idx], regularization = regularization[idx], count_shots = True)
                df_thr_reg = pd.DataFrame(thr_reg_grid, index=delta_list, columns=T_list)

                pd.DataFrame([[f"{res} - threshold (threshold={threshold[idx]}, regularization={regularization[idx]})"]]).to_excel(
                    writer, sheet_name=res, startrow=start, startcol=0, header=False, index=False)
                start += 1
                df_thr_reg.to_excel(writer, sheet_name=res, startrow=start)
                start += len(df_thr_reg) + 3

                if shot_counts_reg is not None:
                    df_shots_reg = pd.DataFrame(shot_counts_reg, index=delta_list, columns=T_list)
                    df_shots_reg.index.name = "delta"
                    df_shots_reg.columns.name = "T"

                    # Place shot grid directly under the threshold grid
                    pd.DataFrame([[f"{res} - remaining shots under threshold, with regularization"]]).to_excel(
                        writer, sheet_name=res, startrow=start, startcol=0, header=False, index=False
                    )
                    start += 1
                    df_shots_reg.to_excel(writer, sheet_name=res, startrow=start)

    return out_path