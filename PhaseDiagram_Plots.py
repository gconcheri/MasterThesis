import numpy as np
import matplotlib.pyplot as plt
import copy

import PhaseDiagram as PD
import os



def plot_single_entry_fromdatagrid(
    data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles, name="op_real",
    figsize=(12, 5), shot_idx=None, regularization = None, threshold=None, log_list=None, layout="row", color_list=None,
    save=False, save_dir_image=None, filename=None,
    ax=None,  # new: pass an existing axis (only when name is a single string)
    axes=None,  # new: pass external axes array when name is a list
    show=True   # new: control plt.show()
): #color_list could be plt.cm.tab10.colors or similar
    """
    Plots a single entry from the data grid for given delta and T indices.
    If name is a list, creates subplots for each specified quantity.
    If log_list is a list of booleans, it specifies whether to plot the log for each corresponding quantity in name.
    layout: "row", "col", or (nrows, ncols)
    If log_bool = True for the loop_e, then we place absolute value before taking log to avoid issues with negative values.

    If ax or axes provided, drawing happens there (no new figure, no show unless show=True at caller end).
    When embedding inside a larger grid (combined plots), pass a single name (string) and an ax.
    """
    #data_grid = copy.deepcopy(data_grid)  # Add this line to avoid modifying the original data_grid
    #I don't need it because I put this line inside the get_regularized_data_grid and remove_shots_fromdatagrid functions

    if not isinstance(name, list):
        single_name_mode = True
    else:
        single_name_mode = False

    if ax is not None and not single_name_mode:
        raise ValueError("When providing ax, 'name' must be a single string (not a list). Use axes for multiple names.")
    if axes is not None and single_name_mode:
        raise ValueError("When providing axes, 'name' must be a list.")

    # Accept a single bool for single metric too
    single_log_mode = False
    if single_name_mode and isinstance(log_list, bool):
        single_log_mode = log_list
    elif single_name_mode and log_list is not None:
        raise ValueError("For single metric, log_list must be a single bool or None.")
    elif not single_name_mode and log_list is not None:
        if len(log_list) != len(name):
            raise ValueError("log_list length must match name list length.")

    if regularization is not None:
        data_grid = PD.get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)
    if threshold is not None:
        data_grid = PD.remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)
    
    entry = data_grid[delta_idx, T_idx]

    if entry is None:
        return print(f"No data available for Δ = {delta_list[delta_idx]}, T = {T_list[T_idx]}")
    
    floquet_cycles = np.arange(N_cycles + 1)
    freqs = PD.frequencies(N_cycles + 1)


    def _plot_shots(data, x, title, ylabel, axx, log_bool=False, color_list=None, is_ft=False):
        if shot_idx is not None:
            if 0 <= shot_idx < len(data):
                ydata = data[shot_idx]
                axx.plot(x, ydata)
            else:
                raise ValueError(f"shot_idx {shot_idx} out of range (max {len(data)-1})") 
        else:
            nshots = len(data)
            if color_list is None:
                color_list = [plt.cm.viridis(i / max(1, nshots-1 if nshots>1 else 1)) for i in range(nshots)]
            for i, shot in enumerate(data):
                axx.plot(x, shot, color=color_list[i], alpha=0.85)
        axx.set_xlabel('Frequency' if is_ft else 'Floquet Cycles')
        axx.set_ylabel(("Log " if log_bool else "") + ylabel)
        if log_bool:
            axx.set_yscale("log")
        axx.set_title(title)

    #Determine layout
    def _prep_axes_for_names(nplots, figsize, layout):
        if axes is not None:
            return axes, None  # external
        if layout == "row":
            fig, axs = plt.subplots(1, nplots, figsize=(figsize[0]*nplots, figsize[1]))
        elif layout == "col":
            fig, axs = plt.subplots(nplots, 1, figsize=(figsize[0], figsize[1]*nplots))
        elif isinstance(layout, tuple) and len(layout) == 2:
            nrows, ncols = layout
            if nrows * ncols < nplots:
                raise ValueError("Layout grid too small for number of subplots.")
            fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
            axs = axs.flatten()
        else:
            raise ValueError("layout must be 'row', 'col', or a tuple (nrows, ncols)")
        return axs if isinstance(axs, np.ndarray) else np.array([axs]), fig

    # Map metadata
    title_map = {
        "op_real": ("Order Parameter", r"$\eta$", False),
        "loop_0": ("Loop Expectation Value (no anyon)", r"$\langle O_0 \rangle$", False),
        "loop_e": ("Loop Expectation Value (e anyon)", r"$\langle O_e \rangle$", False),
        "op_ft": ("Fourier Transform", r"$|FT(\eta)|$", True),
    }

    if single_name_mode:
        # Single metric (possibly inside combined grid)
        metric = name
        if metric not in title_map:
            return
        title, ylabel, is_ft = title_map[metric]
        log_bool = single_log_mode
        # Choose axis
        internal_fig = None
        target_ax = ax
        if target_ax is None:
            internal_fig, target_ax = plt.subplots(figsize=figsize)
        data = entry[metric]
        if metric == "op_ft":
            if isinstance(data, list):
                mag = [np.abs(s) for s in data]
                _plot_shots(mag, freqs, title, ylabel, target_ax, log_bool=log_bool, is_ft=True)
            else:
                target_ax.plot(freqs, np.abs(data))
                target_ax.set_xlabel("Frequency")
                target_ax.set_ylabel(ylabel)
                target_ax.set_title(title)
        else:
            if isinstance(data, list) and delta_idx != 0:
                _plot_shots(data, floquet_cycles, title, ylabel, target_ax, log_bool=log_bool)
            else:
                target_ax.plot(floquet_cycles, data)
                target_ax.set_xlabel("Floquet Cycles")
                target_ax.set_ylabel(ylabel)
                target_ax.set_title(title)
        if ax is None:
            target_ax.set_title(f"{title}\nΔ={delta_list[delta_idx]:.3f}, T={T_list[T_idx]:.3f}")
        if save and ax is None:
            if save_dir_image is None:
                save_dir_image = os.path.join("figures_phasediagram", "single_entries")
            os.makedirs(save_dir_image, exist_ok=True)
            if filename is None:
                filename = f"single_entry_delta_{delta_list[delta_idx]:.3f}_T_{T_list[T_idx]:.3f}_{metric}.svg"
            internal_fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
        if show and ax is None:
            plt.show()
        return

    # Multiple metrics case
    nplots = len(name)
    axs, internal_fig = _prep_axes_for_names(nplots, figsize, layout)

    for idx, metric in enumerate(name):
        if metric not in title_map:
            axs[idx].set_axis_off()
            continue
        title, ylabel, is_ft = title_map[metric]
        log_bool = log_list[idx] if log_list is not None else False
        data = entry[metric]
        if metric == "op_ft":
            if isinstance(data, list) and delta_idx != 0:
                mag = [np.abs(s) for s in data]
                _plot_shots(mag, freqs, title, ylabel, axs[idx], log_bool=log_bool, is_ft=True)
            else:
                axs[idx].plot(freqs, np.abs(data))
                axs[idx].set_xlabel("Frequency")
                axs[idx].set_ylabel(("Log " if log_bool else "") + ylabel)
                if log_bool:
                    axs[idx].set_yscale("log")
                axs[idx].set_title(title)
        else:
            if isinstance(data, list) and delta_idx != 0:
                if log_bool:
                    data_plot = [np.abs(s) for s in data]
                else:
                    data_plot = data
                _plot_shots(data_plot, floquet_cycles, title, ylabel, axs[idx], log_bool=log_bool, is_ft=False)
            else:
                yvals = np.abs(data) if log_bool else data
                axs[idx].plot(floquet_cycles, yvals)
                axs[idx].set_xlabel("Floquet Cycles")
                axs[idx].set_ylabel(("Log " if log_bool else "") + ylabel)
                if log_bool:
                    axs[idx].set_yscale("log")
                axs[idx].set_title(title)

    if axes is None:
        internal_fig.suptitle(f'Δ = {delta_list[delta_idx]}, T = {T_list[T_idx]}', fontsize=14)
        plt.tight_layout()
        if save:
            if save_dir_image is None:
                save_dir_image = os.path.join("figures_phasediagram", "single_entries")
            os.makedirs(save_dir_image, exist_ok=True)
            if filename is None:
                name_part = "-".join(name)
                filename = f"single_entry_delta_{delta_list[delta_idx]:.3f}_T_{T_list[T_idx]:.3f}_{name_part}.svg"
            internal_fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
        if show:
            plt.show()


def plot_all_T_fixed_delta_fromdatagrid(
    data_grid, delta_idx, T_list, delta_list, N_cycles, name = "op_real", figsize = (12,5),
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_dir_image: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg",
    combined: bool = False, ncols: int | None = None
):
    """
    If combined=False (default): old behavior (one figure per T).
    If combined=True: creates a single figure with subplots for all T at fixed delta.
    Combined mode supports only single metric (name is string).
    """
    if combined and isinstance(name, list):
        raise ValueError("combined=True currently supports only a single metric (name must be a string).")

    if regularization is not None:
        data_grid = PD.get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)
    if threshold is not None:
        data_grid = PD.remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)

    if not combined:
        for T_idx in range(len(T_list)):
            fn = None
            if save:
                name_part = name if isinstance(name, str) else "-".join(name)
                prefix = filename_prefix or f"delta_{delta_list[delta_idx]:.3f}"
                fn = f"{prefix}_T_{T_list[T_idx]:.3f}_{name_part}" + suffix
            plot_single_entry_fromdatagrid(
                data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
                name = name, figsize = figsize, log_list=log_list, layout=layout, color_list=color_list,
                save=save, save_dir_image=save_dir_image, filename=fn
            )
        return

    # Combined mode
    nplots = len(T_list)
    if ncols is None:
        ncols = min(nplots, 4)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols*0.6, figsize[1]*nrows))
    axes = np.array(axes).reshape(-1)

    for T_idx, ax in enumerate(axes[:nplots]):
        plot_single_entry_fromdatagrid(
            data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
            name = name, ax=ax, show=False
        )
        ax.set_title(f"T={T_list[T_idx]:.3f}")
    # Remove unused axes
    for ax in axes[nplots:]:
        ax.set_axis_off()

    fig.suptitle(f"Δ = {delta_list[delta_idx]:.3f} ({name})", fontsize=14)
    plt.tight_layout()

    if save:
        os.makedirs(save_dir_image, exist_ok=True)
        prefix = filename_prefix or f"delta_{delta_list[delta_idx]:.3f}"
        filename = f"{prefix}_ALLT_{name}{suffix}"
        fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
    plt.show()


def plot_all_delta_fixed_T_fromdatagrid(
    data_grid, T_idx, T_list, delta_list, N_cycles, name = "op_real", figsize = (12,5),
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_dir_image: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg",
    combined: bool = False, ncols: int | None = None
):
    """
    Similar combined behavior for fixed T over all deltas.
    """
    if combined and isinstance(name, list):
        raise ValueError("combined=True currently supports only a single metric (name must be a string).")

    if regularization is not None:
        data_grid = PD.get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)
    if threshold is not None:
        data_grid = PD.remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)

    if not combined:
        for delta_idx in range(len(delta_list)):
            fn = None
            if save:
                name_part = name if isinstance(name, str) else "-".join(name)
                prefix = filename_prefix or f"T_{T_list[T_idx]:.3f}"
                fn = f"{prefix}_delta_{delta_list[delta_idx]:.3f}_{name_part}" + suffix
            plot_single_entry_fromdatagrid(
                data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
                name = name, figsize = figsize, log_list=log_list, layout=layout, color_list=color_list,
                save=save, save_dir_image=save_dir_image, filename=fn
            )
        return

    nplots = len(delta_list)
    if ncols is None:
        ncols = min(nplots, 4)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols*0.6, figsize[1]*nrows))
    axes = np.array(axes).reshape(-1)

    for delta_idx, ax in enumerate(axes[:nplots]):
        plot_single_entry_fromdatagrid(
            data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles,
            name = name, ax=ax, show=False
        )
        ax.set_title(f"Δ={delta_list[delta_idx]:.3f}")
    for ax in axes[nplots:]:
        ax.set_axis_off()

    fig.suptitle(f"T = {T_list[T_idx]:.3f} ({name})", fontsize=14)
    plt.tight_layout()
    if save:
        os.makedirs(save_dir_image, exist_ok=True)
        prefix = filename_prefix or f"T_{T_list[T_idx]:.3f}"
        filename = f"{prefix}_ALLDELTA_{name}{suffix}"
        fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
    plt.show()


def plot_all_entries_fromdatagrid(
    data_grid, T_list, delta_list, N_cycles = 10, name = "op_real", figsize = (12,5),
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_dir_image: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg",
    combined: bool = False, ncols: int | None = None
):
    """
    If combined=False: old behavior (many individual figures).
    If combined=True: single large grid of subplots (requires name to be a single metric).
    """
    if combined and isinstance(name, list):
        raise ValueError("combined=True currently supports only a single metric (name must be a string).")

    data_grid = copy.deepcopy(data_grid)
    if regularization is not None:
        data_grid = PD.get_regularized_data_grid(data_grid, T_list, delta_list, regularization=regularization)
    if threshold is not None:
        data_grid = PD.remove_shots_fromdatagrid(data_grid, T_list, delta_list, threshold=threshold)

    if not combined:
        for delta_idx in range(len(delta_list)):
            for T_idx in range(len(T_list)):
                fn = None
                if save:
                    name_part = name if isinstance(name, str) else "-".join(name)
                    prefix = filename_prefix or "all_entries"
                    fn = f"{prefix}_delta_{delta_list[delta_idx]:.3f}_T_{T_list[T_idx]:.3f}_{name_part}" + suffix
                plot_single_entry_fromdatagrid(
                    data_grid, delta_idx, T_idx, T_list, delta_list, N_cycles, name = name, figsize = figsize,
                    log_list=log_list, layout=layout, color_list=color_list,
                    save=save, save_dir_image=save_dir_image, filename=fn
                )
        return

    # Combined grid
    n_delta = len(delta_list)
    n_T = len(T_list)
    if ncols is not None:
        raise ValueError("ncols is not used in full grid mode; layout is fixed to (len(delta), len(T)).")
    fig, axes = plt.subplots(n_delta, n_T, figsize=(figsize[0]*n_T*0.5, figsize[1]*n_delta*0.5), squeeze=False)

    for i in range(n_delta):
        for j in range(n_T):
            ax = axes[i, j]
            plot_single_entry_fromdatagrid(
                data_grid, i, j, T_list, delta_list, N_cycles,
                name=name, ax=ax, show=False
            )
            if i == n_delta - 1:
                ax.set_xlabel(f"T={T_list[j]:.3f}")
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(f"Δ={delta_list[i]:.3f}")
            else:
                ax.set_ylabel("")
    fig.suptitle(f"All entries ({name})", fontsize=14)
    plt.tight_layout()
    if save:
        os.makedirs(save_dir_image, exist_ok=True)
        filename = (filename_prefix or "all_entries_grid") + f"_{name}{suffix}"
        fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
    plt.show()

# ...existing code...

def plot_entries_grid_fromdatagrid(
    data_grid, selected_delta_list, selected_T_list, T_list, delta_list, N_cycles,
    name="op_real", figsize=(12,5), regularization=None, threshold=None,
    log_list=None,  # new: allow passing per-metric or single log flag
    metrics_layout=None,  # new: (rows, cols) for metrics inside each cell (when name is a list)
    save=False, save_dir_image=None, filename=None, suffix=".svg"
):
    """
    Build a subplot grid for the cartesian product of selected_delta_list x selected_T_list.

    If name is a string: each cell shows that metric (original behavior).
    If name is a list: each (delta, T) cell contains a mini-grid of metrics
        defined by metrics_layout (rows, cols). If metrics_layout is None,
        a near-square layout is auto-chosen.

    log_list is forwarded to plot_single_entry_fromdatagrid (bool, list, or None).
    """
    from matplotlib import gridspec  # local import to avoid global dependency if unused

    multi_metric = isinstance(name, list)

    # Map indices (raise if value not found)
    try:
        delta_indices = [delta_list.index(d) for d in selected_delta_list]
        T_indices = [T_list.index(t) for t in selected_T_list]
    except ValueError as e:
        raise ValueError(f"Value in selected lists not found in delta_list/T_list: {e}")

    dg = data_grid
    if regularization is not None:
        dg = PD.get_regularized_data_grid(dg, T_list, delta_list, regularization=regularization)
    if threshold is not None:
        dg = PD.remove_shots_fromdatagrid(dg, T_list, delta_list, threshold=threshold)

    n_delta = len(delta_indices)
    n_T = len(T_indices)

    if not multi_metric:
        # Original (single metric per cell)
        fig, axes = plt.subplots(n_delta, n_T,
                                 figsize=(figsize[0]*n_T*0.5, figsize[1]*n_delta*0.5),
                                 squeeze=False)
        for i, di in enumerate(delta_indices):
            for j, tj in enumerate(T_indices):
                ax = axes[i, j]
                plot_single_entry_fromdatagrid(
                    dg, di, tj, T_list, delta_list, N_cycles,
                    name=name, ax=ax, show=False, log_list=log_list
                )
                if i == n_delta - 1:
                    ax.set_xlabel(f"T={selected_T_list[j]:.3f}")
                else:
                    ax.set_xlabel("")
                if j == 0:
                    ax.set_ylabel(f"Δ={selected_delta_list[i]:.3f}")
                else:
                    ax.set_ylabel("")
        fig.suptitle(f"Selected entries grid ({name})", fontsize=14)
        plt.tight_layout()
        if save:
            if save_dir_image is None:
                save_dir_image = os.path.join("figures_phasediagram", "selected_grids")
            os.makedirs(save_dir_image, exist_ok=True)
            if filename is None:
                filename = f"selected_grid_{name}{suffix}"
            fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
        plt.show()
        return fig, axes

    # Multi-metric case: prepare outer grid, then sub-grids per cell
    n_metrics = len(name)
    if metrics_layout is None:
        # auto layout (square-ish)
        rows = int(np.floor(np.sqrt(n_metrics)))
        while rows > 0 and n_metrics % rows != 0:
            rows -= 1
        if rows == 0:
            rows = 1
        cols = n_metrics // rows
        if rows * cols < n_metrics:
            # fallback
            rows = int(np.ceil(np.sqrt(n_metrics)))
            cols = int(np.ceil(n_metrics / rows))
        metrics_layout = (rows, cols)
    inner_rows, inner_cols = metrics_layout
    if inner_rows * inner_cols < n_metrics:
        raise ValueError("metrics_layout grid too small for number of metrics.")

    # Scale figure size: per (delta,T) block smaller than full figsize
    cell_w = figsize[0] / max(2, 2)  # simple heuristic
    cell_h = figsize[1] / max(2, 2)
    fig = plt.figure(figsize=(cell_w * n_T * inner_cols * 0.8,
                              cell_h * n_delta * inner_rows * 0.8))
    outer = gridspec.GridSpec(n_delta, n_T, figure=fig, wspace=0.6, hspace=0.8)

    # Keep references to inner axes: axes_grid[i][j] -> list of axes for that cell
    axes_grid = [[None]*n_T for _ in range(n_delta)]

    for i, di in enumerate(delta_indices):
        for j, tj in enumerate(T_indices):
            sub = gridspec.GridSpecFromSubplotSpec(inner_rows, inner_cols,
                                                   subplot_spec=outer[i, j],
                                                   wspace=0.35, hspace=0.4)
            # Create all inner axes
            inner_axes = []
            for m in range(inner_rows * inner_cols):
                if m < n_metrics:
                    ax_m = fig.add_subplot(sub[m // inner_cols, m % inner_cols])
                    inner_axes.append(ax_m)
                else:
                    # extra unused cell -> create and hide
                    ax_m = fig.add_subplot(sub[m // inner_cols, m % inner_cols])
                    ax_m.set_axis_off()
                    inner_axes.append(ax_m)

            # Pass axes (flattened) to the existing multi-metric plotter
            plot_single_entry_fromdatagrid(
                dg, di, tj, T_list, delta_list, N_cycles,
                name=name, axes=np.array(inner_axes), show=False, log_list=log_list
            )

            # Annotate only on first internal axis
            label_ax = inner_axes[0]
            if i == n_delta - 1:
                label_ax.set_xlabel(f"T={selected_T_list[j]:.3f}")
            else:
                label_ax.set_xlabel("")
            if j == 0:
                label_ax.set_ylabel(f"Δ={selected_delta_list[i]:.3f}")
            else:
                label_ax.set_ylabel("")

            axes_grid[i][j] = inner_axes

    fig.suptitle(f"Selected entries grid (metrics: {', '.join(name)})", fontsize=14)
    plt.tight_layout()
    if save:
        if save_dir_image is None:
            save_dir_image = os.path.join("figures_phasediagram", "selected_grids")
        os.makedirs(save_dir_image, exist_ok=True)
        if filename is None:
            filename = f"selected_grid_{'-'.join(name)}{suffix}"
        fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
    plt.show()
    return fig, axes_grid

# ...existing code...

def plot_all_T_fixed_delta(
    delta, T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10,
    name = "op_real", figsize = (12,5), regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg",
    combined: bool = False, ncols: int | None = None
):
    data_grid = PD.load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    if delta in delta_list:
        delta_idx = delta_list.index(delta)
        save_dir_image = None
        if save:
            if save_subdir is None:
                save_subdir = "single_entries"
                save_subdir = save_subdir + create_suffix(log = log_list, thresh = threshold, reg = regularization)
                save_subdir = os.path.join("figures", save_subdir)
            save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
            os.makedirs(save_dir_image, exist_ok=True)

        plot_all_T_fixed_delta_fromdatagrid(
            data_grid, delta_idx, T_list, delta_list, N_cycles, name = name, figsize = figsize,
            log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename_prefix=filename_prefix, suffix=suffix,
            combined=combined, ncols=ncols
        )
    else:
        print("Provided delta not in the respective list.")
        return

def plot_all_delta_fixed_T(
    T, T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10, name = "op_real", figsize = (12,5), 
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg",
    combined: bool = False, ncols: int | None = None
):
    data_grid = PD.load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    if T in T_list:
        T_idx = T_list.index(T)
        save_dir_image = None
        if save:
            if save_subdir is None:
                save_subdir = "single_entries"
                save_subdir = save_subdir + create_suffix(log = log_list, thresh = threshold, reg = regularization)
                save_subdir = os.path.join("figures", save_subdir)

            save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
            os.makedirs(save_dir_image, exist_ok=True)

        plot_all_delta_fixed_T_fromdatagrid(
            data_grid, T_idx, T_list, delta_list, N_cycles, name = name, figsize = figsize, 
            regularization = regularization, threshold=threshold, log_list=log_list, layout=layout, color_list=color_list,
            save=save, save_dir_image=save_dir_image, filename_prefix=filename_prefix, suffix=suffix,
            combined=combined, ncols=ncols
        )
    else:
        print("Provided T not in the respective list.")
        return

def plot_all_entries(
    T_list, delta_list, save_dir, general_dir = "phasediagram", N_cycles = 10, name = "op_real", figsize = (12,5), 
    regularization = None, threshold=None, log_list = None, layout="row", color_list=None,
    save: bool = False, save_subdir: str | None = None, filename_prefix: str | None = None, suffix: str = ".svg",
    combined: bool = False
):
    data_grid = PD.load_saved_results(T_list, delta_list, save_dir, general_dir = general_dir)
    save_dir_image = None
    if save:
        if save_subdir is None:
            save_subdir = os.path.join("figures", "single_entries")
        save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
        os.makedirs(save_dir_image, exist_ok=True)

    plot_all_entries_fromdatagrid(
        data_grid, T_list, delta_list, N_cycles, name = name, figsize = figsize, 
        regularization = regularization, threshold=threshold, log_list=log_list, layout=layout, color_list=color_list,
        save=save, save_dir_image=save_dir_image, filename_prefix=filename_prefix, suffix=suffix,
        combined=combined
    )

# ...existing code...

def plot_entries_grid(
    selected_delta_list, selected_T_list, T_list, delta_list, save_dir,
    general_dir="phasediagram", N_cycles=10, name="op_real", figsize=(12,5),
    regularization=None, threshold=None, log_list=None, metrics_layout=None,
    save=False, save_subdir: str | None = None, filename: str | None = None, suffix: str = ".svg"
):
    """
    Wrapper that loads data from disk and calls plot_entries_grid_fromdatagrid.

    selected_delta_list, selected_T_list: subsets to plot (values must be in delta_list / T_list).
    name: str for single metric or list[str] for multiple metrics per cell.
    metrics_layout: (rows, cols) for the mini-grid inside each cell when name is a list. If None -> auto.
    """
    data_grid = PD.load_saved_results(T_list, delta_list, save_dir, general_dir=general_dir)

    save_dir_image = None
    if save:
        if save_subdir is None:
            # keep it separate from single_entries so files are organized
            save_subdir = os.path.join("figures", "selected_grids")
        save_dir_image = os.path.join(general_dir, save_dir, save_subdir)
        os.makedirs(save_dir_image, exist_ok=True)

    fig, axes = plot_entries_grid_fromdatagrid(
        data_grid=data_grid,
        selected_delta_list=selected_delta_list,
        selected_T_list=selected_T_list,
        T_list=T_list,
        delta_list=delta_list,
        N_cycles=N_cycles,
        name=name,
        figsize=figsize,
        regularization=regularization,
        threshold=threshold,
        log_list=log_list,
        metrics_layout=metrics_layout,
        save=save,
        save_dir_image=save_dir_image,
        filename=filename,
        suffix=suffix
    )
    return fig, axes

def create_suffix(**kwargs):
    """
    Build a suffix string for folder or filename based on provided keyword arguments.
    - If a value is a list and contains True, adds just the key (e.g., 'log').
    - If a value is not None and not False, adds key+value (e.g., 'thresh2').
    - Skips None and False values.
    """
    parts = []
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        if isinstance(value, list):
            # Special case for lists of bools (like log_list)
            if any(value):
                parts.append(str(key))
        else:
            parts.append(f"{key}{value}")
    return ("_" + "_".join(parts)) if parts else ""

# def plot_entries_grid_fromdatagrid(
#     data_grid, selected_delta_list, selected_T_list, T_list, delta_list, N_cycles,
#     name="op_real", figsize=(12,5), regularization=None, threshold=None,
#     save=False, save_dir_image=None, filename=None, suffix=".svg"
# ):
#     """
#     New: Build a subplot grid for the cartesian product of selected_delta_list x selected_T_list.
#     name must be a single metric.
#     """
#     if isinstance(name, list):
#         raise ValueError("This helper supports only a single metric for each cell.")
#     # Map indices
#     delta_indices = [delta_list.index(d) for d in selected_delta_list]
#     T_indices = [T_list.index(t) for t in selected_T_list]

#     dg = data_grid
#     if regularization is not None:
#         dg = PD.get_regularized_data_grid(dg, T_list, delta_list, regularization=regularization)
#     if threshold is not None:
#         dg = PD.remove_shots_fromdatagrid(dg, T_list, delta_list, threshold=threshold)

#     n_delta = len(delta_indices)
#     n_T = len(T_indices)
#     fig, axes = plt.subplots(n_delta, n_T, figsize=(figsize[0]*n_T*0.5, figsize[1]*n_delta*0.5), squeeze=False)

#     for i, di in enumerate(delta_indices):
#         for j, tj in enumerate(T_indices):
#             ax = axes[i, j]
#             plot_single_entry_fromdatagrid(
#                 dg, di, tj, T_list, delta_list, N_cycles,
#                 name=name, ax=ax, show=False
#             )
#             if i == n_delta - 1:
#                 ax.set_xlabel(f"T={selected_T_list[j]:.3f}")
#             else:
#                 ax.set_xlabel("")
#             if j == 0:
#                 ax.set_ylabel(f"Δ={selected_delta_list[i]:.3f}")
#             else:
#                 ax.set_ylabel("")
#     fig.suptitle(f"Selected entries grid ({name})", fontsize=14)
#     plt.tight_layout()
#     if save:
#         if save_dir_image is None:
#             save_dir_image = os.path.join("figures_phasediagram", "selected_grids")
#         os.makedirs(save_dir_image, exist_ok=True)
#         if filename is None:
#             filename = f"selected_grid_{name}{suffix}"
#         fig.savefig(os.path.join(save_dir_image, filename), bbox_inches="tight")
#     plt.show()
#     return fig, axes

#Example usage

# Single metric (same as before)
# plot_entries_grid_fromdatagrid(data_grid,
#     selected_delta_list=[0.0,0.2,0.4],
#     selected_T_list=[0.3,0.6],
#     T_list=T_list, delta_list=delta_list, N_cycles=10,
#     name="op_real")

# # Multiple metrics (auto layout)
# plot_entries_grid_fromdatagrid(data_grid,
#     selected_delta_list=[0.0,0.2],
#     selected_T_list=[0.3,0.6,0.9],
#     T_list=T_list, delta_list=delta_list, N_cycles=10,
#     name=["op_real","loop_0","loop_e","op_ft"],
#     metrics_layout=(2,2), log_list=[False, False, False, False])