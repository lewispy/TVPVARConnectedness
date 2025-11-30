import numpy as np

class TVPVARConnectedness:
    """
    Time-Varying Parameter VAR with forgetting factors + Diebold-Yilmaz
    connectedness measures following Antonakakis et al. (2020).
    """

    def __init__(self, y, lags=1, horizon=10, kappa1=0.99, kappa2=0.96,
                 prior_length=None, dates=None):
        """
        Parameters
        ----------
        y : array-like, shape (T, m)
            Data matrix with T observations of m variables.
        lags : int
            VAR lag order p.
        horizon : int
            Forecast horizon H for GFEVD / connectedness.
        kappa1 : float
            Forgetting factor for state covariance (0 < kappa1 <= 1).
        kappa2 : float
            Forgetting factor for shock covariance (0 < kappa2 <= 1).
        prior_length : int or None
            Number of initial observations used to estimate the fixed-parameter VAR
            prior. If None, it is set to max(60, 5 * m) but not more than T - p - 1.
        dates : array-like or None
            Optional time index of length T (for pandas output).
        """
        self.y = np.asarray(y)
        if self.y.ndim != 2:
            raise ValueError("y must be 2D array (T, m).")
        self.T, self.m = self.y.shape
        self.p = int(lags)
        self.H = int(horizon)
        self.k1 = float(kappa1)
        self.k2 = float(kappa2)
        if prior_length is None:
            prior_length = min(self.T - self.p - 1, max(60, 5 * self.m))
        self.prior_length = prior_length
        self.dates = np.asarray(dates) if dates is not None else None

        # Placeholders
        self.alpha_t = None  # state vectors over time (T_eff, n_state)
        self.P_t = None      # state covariance matrices (T_eff, n_state, n_state)
        self.Sigma_t = None  # innovation covariances (T_eff, m, m)
        self.A_t = None      # coefficient matrices (T_eff, m, m*p)
        self.phi_t = None    # GFEVD matrices (T_eff, m, m)
        self.tci = None      # total connectedness index (T_eff,)
        self.to_dir = None   # directional TO others (T_eff, m)
        self.from_dir = None # directional FROM others (T_eff, m)
        self.net_dir = None  # net directional (T_eff, m)
        self.npdc = None     # net pairwise connectedness (T_eff, m, m)

    # ---------- Plot layout helper ----------

    @staticmethod
    def _get_plot_dimension(dim: int):
        """
        Return (rows, cols) for arranging `dim` plots in a grid that is
        as square as possible, with total cells >= dim. Guarantees rows >= cols.
        Example: 15 -> (4, 4), 13 -> (4, 4), 20 -> (5, 4).
        """
        import math

        if not isinstance(dim, int):
            raise TypeError("dim must be an int")
        if dim < 1:
            raise ValueError("dim must be >= 1")

        k = math.isqrt(dim)
        if k * k < dim:
            k += 1

        rows = k
        cols = (dim + rows - 1) // rows  # ceil(dim / rows)
        return rows, cols


    # ---------- Helpers for the prior ----------

    def _build_var_matrices(self, start_idx, end_idx):
        """
        Build lagged regressor matrix Z and response matrix Y for VAR(p)
        using observations from indices [start_idx, end_idx).
        """
        y = self.y
        p = self.p
        m = self.m

        T_sub = end_idx - start_idx
        if T_sub <= p:
            raise ValueError("Not enough observations in the selected range for VAR lags.")
        n_rows = T_sub - p

        Y = np.empty((n_rows, m))
        Z = np.empty((n_rows, m * p))

        row = 0
        for t in range(start_idx + p, end_idx):
            Y[row, :] = y[t, :]
            z_t = []
            for lag in range(1, p + 1):
                z_t.append(y[t - lag, :])
            Z[row, :] = np.concatenate(z_t)
            row += 1

        return Y, Z

    def _ols_prior(self):
        """
        Fixed-parameter VAR(p) on the first `prior_length` observations.

        Returns
        -------
        alpha0 : (n_state,)  initial state vec(A_ols)
        P0     : (n_state,n_state) initial state covariance
        Sigma0 : (m,m)      initial innovation covariance
        """
        m = self.m
        p = self.p
        prior_end = self.prior_length

        Y, Z = self._build_var_matrices(0, prior_end)
        ZZ = Z.T @ Z
        ZZ_inv = np.linalg.inv(ZZ + 1e-8 * np.eye(ZZ.shape[0]))  # small ridge
        B = ZZ_inv @ Z.T @ Y            # (m*p, m)

        A0 = B.T                        # (m, m*p)
        alpha0 = A0.reshape(-1, order="C")

        Y_hat = Z @ B
        E = Y - Y_hat
        Sigma0 = np.cov(E.T, bias=False)

        n_state = m * m * p
        P0 = 0.01 * np.eye(n_state)     # diffuse but finite prior

        return alpha0, P0, Sigma0

    # ---------- Main estimation ----------

    def fit(self):
        """
        Run TVP-VAR Kalman filter with discount factors and compute
        time-varying connectedness measures.
        """
        y = self.y
        T, m, p = self.T, self.m, self.p
        H = self.H
        k1, k2 = self.k1, self.k2

        alpha0, P0, Sigma0 = self._ols_prior()
        n_state = alpha0.shape[0]

        start_t = self.prior_length
        if start_t <= p:
            raise ValueError("prior_length must be greater than VAR lag order.")

        T_eff = T - start_t

        alpha_t = np.zeros((T_eff, n_state))
        P_t = np.zeros((T_eff, n_state, n_state))
        Sigma_t = np.zeros((T_eff, m, m))
        A_t = np.zeros((T_eff, m, m * p))

        # initial predictions
        alpha_pred = alpha0.copy()
        P_pred = P0.copy()
        Sigma_prev = Sigma0.copy()

        # selection matrix J for VMA representation
        mp = m * p
        J = np.zeros((mp, m))
        J[0:m, :] = np.eye(m)

        for idx, t in enumerate(range(start_t, T)):
            # lagged regressor z_{t-1}
            z_list = []
            for lag in range(1, p + 1):
                z_list.append(y[t - lag, :])
            z_t = np.concatenate(z_list)     # (mp,)

            # design matrix: y_t = (I_m ⊗ z_t') alpha_t + e_t
            Z_t = np.kron(np.eye(m), z_t.reshape(1, -1))  # (m, n_state)

            # 1. state prediction with discount k1
            alpha_pred = alpha_pred
            P_pred = P_pred / k1

            # 2. update shock covariance with discount k2
            y_t = y[t, :].reshape(-1, 1)
            y_hat_pred = (Z_t @ alpha_pred).reshape(-1, 1)
            e_t = y_t - y_hat_pred                 # (m,1)
            Sigma_curr = k2 * Sigma_prev + (1.0 - k2) * (e_t @ e_t.T)

            # 3. forecast error covariance
            F_t = Z_t @ P_pred @ Z_t.T + Sigma_curr
            F_t_reg = F_t + 1e-8 * np.eye(m)

            # 4. Kalman gain
            K_t = P_pred @ Z_t.T @ np.linalg.inv(F_t_reg)  # (n_state,m)

            # 5. state update
            alpha_upd = alpha_pred + (K_t @ e_t).reshape(-1)
            P_upd = (np.eye(n_state) - K_t @ Z_t) @ P_pred

            # store
            alpha_t[idx, :] = alpha_upd
            P_t[idx, :, :] = P_upd
            Sigma_t[idx, :, :] = Sigma_curr
            A_curr = alpha_upd.reshape(m, m * p, order="C")
            A_t[idx, :, :] = A_curr

            # prepare for next step
            alpha_pred = alpha_upd
            P_pred = P_upd
            Sigma_prev = Sigma_curr

        self.alpha_t = alpha_t
        self.P_t = P_t
        self.Sigma_t = Sigma_t
        self.A_t = A_t

        # connectedness
        self._compute_connectedness(J)
        return self

    # ---------- GFEVD and connectedness ----------

    def _compute_gfevd_single(self, A, Sigma, J):
        """
        GFEVD matrix for one time t.
        """
        m, mp = A.shape
        p = mp // m
        H = self.H

        # companion matrix
        if p == 1:
            M = A
        else:
            M_top = A
            bottom_left = np.eye(m * (p - 1))
            bottom_right = np.zeros((m * (p - 1), m))
            M_bottom = np.hstack([bottom_left, bottom_right])
            M = np.vstack([M_top, M_bottom])

        # powers of M and B_h = J' M^h J
        B_list = np.zeros((H, m, m))
        M_power = np.eye(mp)
        for h in range(H):
            B_h = J.T @ M_power @ J
            B_list[h, :, :] = B_h
            M_power = M_power @ M

        # GIRFs and GFEVD
        psi = np.zeros((H, m, m))  # h, i, j

        for h in range(H):
            B_h = B_list[h, :, :]
            for j in range(m):
                sigma_jj = Sigma[j, j]
                if sigma_jj <= 0:
                    continue
                col_j = Sigma[:, j]
                psi[h, :, j] = (B_h @ col_j) / np.sqrt(sigma_jj)

        phi = np.zeros((m, m))
        for i in range(m):
            numer_ij = (psi[:, i, :] ** 2).sum(axis=0)
            denom_i = numer_ij.sum()
            if denom_i <= 0:
                phi[i, :] = np.nan
            else:
                phi[i, :] = numer_ij / denom_i

        return phi

    def _compute_connectedness(self, J):
        """
        Time-varying GFEVDs and connectedness indices.
        """
        if self.A_t is None or self.Sigma_t is None:
            raise RuntimeError("Run fit() first.")

        T_eff = self.A_t.shape[0]
        m = self.m

        phi_t = np.zeros((T_eff, m, m))
        tci = np.zeros(T_eff)
        to_dir = np.zeros((T_eff, m))
        from_dir = np.zeros((T_eff, m))
        net_dir = np.zeros((T_eff, m))
        npdc = np.zeros((T_eff, m, m))

        for t_idx in range(T_eff):
            A = self.A_t[t_idx, :, :]
            Sigma = self.Sigma_t[t_idx, :, :]

            phi = self._compute_gfevd_single(A, Sigma, J)

            # make sure rows sum to 1
            row_sums = phi.sum(axis=1, keepdims=True)
            phi = np.where(row_sums > 0, phi / row_sums, np.nan)

            phi_t[t_idx, :, :] = phi

            # total connectedness index
            tci[t_idx] = 100.0 * (1.0 - np.trace(phi) / m)

            # directional FROM (rows) and TO (cols)
            from_i = 100.0 * (phi.sum(axis=1) - np.diag(phi))
            to_i = 100.0 * (phi.sum(axis=0) - np.diag(phi))

            from_dir[t_idx, :] = from_i
            to_dir[t_idx, :] = to_i
            net_dir[t_idx, :] = to_i - from_i

            # net pairwise
            npdc[t_idx, :, :] = 100.0 * (phi.T - phi)

        self.phi_t = phi_t
        self.tci = tci
        self.to_dir = to_dir
        self.from_dir = from_dir
        self.net_dir = net_dir
        self.npdc = npdc

    # ---------- Convenience getters (pandas) ----------

    def get_tci(self):
        import pandas as pd
        idx = self.dates[self.prior_length:] if self.dates is not None else None
        return pd.Series(self.tci, index=idx, name="TCI")

    def get_directional(self):
        import pandas as pd
        idx = self.dates[self.prior_length:] if self.dates is not None else None
        cols = [f"var_{i+1}" for i in range(self.m)]
        to_df = pd.DataFrame(self.to_dir, index=idx, columns=cols)
        from_df = pd.DataFrame(self.from_dir, index=idx, columns=cols)
        net_df = pd.DataFrame(self.net_dir, index=idx, columns=cols)
        return to_df, from_df, net_df

    def get_static_table(self):
        """
        Average connectedness table across time.
        """
        phi_bar = np.nanmean(self.phi_t, axis=0)
        m = self.m
        tci_bar = 100.0 * (1.0 - np.trace(phi_bar) / m)
        from_i = 100.0 * (phi_bar.sum(axis=1) - np.diag(phi_bar))
        to_i = 100.0 * (phi_bar.sum(axis=0) - np.diag(phi_bar))
        net_i = to_i - from_i
        return {
            "phi_bar": phi_bar,
            "to": to_i,
            "from": from_i,
            "net": net_i,
            "tci": tci_bar,
        }
    
    def get_static_table_df(self, labels=None):
        """
        Build a Diebold–Yilmaz style static connectedness table as a DataFrame.

        Rows:    variables + ["TO", "FROM", "NET"]
        Columns: variables + ["FROM"]

        - Main m x m block: 100 * phi_bar (average GFEVD).
        - Last column ("FROM"): FROM-others for each variable.
        - "TO" row:   TO-others for each variable.
        - "FROM" row: FROM-others for each variable (duplicate, for symmetry).
        - "NET" row:  TO - FROM for each variable; bottom-right cell = TCI.
        """
        import numpy as np
        import pandas as pd

        # use existing dict helper
        st = self.get_static_table()
        phi_bar = st["phi_bar"]          # m x m, shares
        to_i    = st["to"]               # length m, in percent
        from_i  = st["from"]             # length m, in percent
        net_i   = st["net"]              # length m, in percent
        tci     = st["tci"]              # scalar, in percent
        m       = self.m

        if labels is None:
            labels = [f"var_{k+1}" for k in range(m)]

        # scale phi_bar to percent
        phi_pct = 100.0 * phi_bar

        # build empty table
        index_names  = list(labels) + ["TO", "FROM", "NET"]
        column_names = list(labels) + ["FROM"]
        table = pd.DataFrame(
            np.nan,
            index=index_names,
            columns=column_names,
        )

        # main block
        table.loc[labels, labels] = phi_pct

        # FROM column for each variable
        table.loc[labels, "FROM"] = from_i

        # TO / FROM / NET rows
        table.loc["TO",   labels] = to_i
        table.loc["FROM", labels] = from_i
        table.loc["NET",  labels] = net_i

        # bottom-right = TCI
        table.loc["NET", "FROM"] = tci

        return table


    # ---------- Extra getters for pairwise series ----------

    def get_pairwise_directional(self, from_idx, to_idx, labels=None):
        """
        Time series of pairwise directional connectedness
        FROM variable `from_idx` TO variable `to_idx`.

        According to the FEVD convention:
        phi[i,j] = share of variance of i explained by shocks in j.
        So directional FROM j TO i is phi[i,j].
        """
        import pandas as pd

        if self.phi_t is None:
            raise RuntimeError("Run fit() first.")

        phi_ijt = self.phi_t[:, to_idx, from_idx]  # T_eff array
        series_idx = self.dates[self.prior_length:] if self.dates is not None else None

        if labels is None:
            name = f"from_{from_idx+1}_to_{to_idx+1}"
        else:
            name = f"from_{labels[from_idx]}_to_{labels[to_idx]}"

        return pd.Series(100.0 * phi_ijt, index=series_idx, name=name)

    def get_net_pairwise(self, i, j, labels=None):
        """
        Time series of net pairwise directional connectedness between i and j.

        npdc[t,i,j] = 100 * (phi[j,i] - phi[i,j])
        > 0 means j dominates i (net transmitter to i),
        < 0 means i dominates j.
        """
        import pandas as pd

        if self.npdc is None:
            raise RuntimeError("Run fit() first.")

        npdc_ij = self.npdc[:, i, j]
        series_idx = self.dates[self.prior_length:] if self.dates is not None else None

        if labels is None:
            name = f"net_{i+1}_vs_{j+1}"
        else:
            name = f"net_{labels[i]}_vs_{labels[j]}"

        return pd.Series(npdc_ij, index=series_idx, name=name)

    # ---------- Plotting helpers ----------

    def plot_tci(self, ax=None):
        """
        Plot Total Connectedness Index over time (single plot with fill_between).
        """
        import matplotlib.pyplot as plt

        tci = self.get_tci()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        x = tci.index
        y = tci.values
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.plot(x, y)
        ax.fill_between(x, 0, y, alpha=0.3)
        ax.set_title("Total Connectedness Index (TCI)", fontsize=16)
        ax.set_ylabel("Percent")
        ax.grid(False, axis='x')
        ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)
        return ax

    def plot_directional_subplots(self, which="to", labels=None, figsize=(14, 8)):
        """
        Plot directional connectedness as a grid of subplots:
        - which = "to"       : TO others
        - which = "from"     : FROM others
        - which = "net_from" : FROM - TO = net FROM others

        Each variable gets its own subplot with fill_between shading.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        to_df, from_df, net_df_to = self.get_directional()

        if which == "to":
            df_plot = to_df
            title_root = "Directional Connectedness TO Others"
        elif which == "from":
            df_plot = from_df
            title_root = "Directional Connectedness FROM Others"
        elif which == "net_from":
            df_plot = from_df - to_df   # net FROM others
            title_root = "Net Directional Connectedness FROM Others"
        else:
            raise ValueError("which must be 'to', 'from', or 'net_from'.")

        if labels is not None and len(labels) == self.m:
            df_plot = df_plot.copy()
            df_plot.columns = labels
        else:
            labels = list(df_plot.columns)

        n = self.m
        rows, cols = self._get_plot_dimension(n)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(df_plot.columns):
            ax = axes[i]
            s = df_plot[col]
            x = s.index
            y = s.values
            ax.axhline(0.0, linestyle="--", linewidth=0.8)
            ax.plot(x, y)
            ax.fill_between(x, 0, y, alpha=0.3)
            ax.set_title(col, fontsize=14)
            ax.grid(False, axis='x')
            ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

        # hide unused axes if grid is larger than m
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(title_root, fontsize=16)
        fig.tight_layout()
        return fig, axes

    def plot_pairwise_directional_subplots(
        self,
        which: str = "j_to_i",
        labels=None,
        figsize=(16, 10),
    ):
        """
        Plot pairwise directional connectedness in subplots.

        Parameters
        ----------
        which : {"j_to_i", "i_to_j", "both"}
            - "j_to_i": for each unordered pair (i, j), plot FROM j TO i
                        i.e. 100 * phi[i, j] as blue line + blue fill.
            - "i_to_j": for each unordered pair (i, j), plot FROM i TO j
                        i.e. 100 * phi[j, i] as blue line + blue fill.
            - "both"  : plot both directions in each subplot:
                        j→i as blue shaded series, i→j as red line only.
        labels : list of str or None
            Optional variable names (length = number of variables).
        figsize : tuple
            Figure size passed to plt.subplots.

        Number of subplots = (N^2 - N)/2 where N is number of variables.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from itertools import combinations

        if which not in ("j_to_i", "i_to_j", "both"):
            raise ValueError("which must be 'j_to_i', 'i_to_j', or 'both'.")

        if self.phi_t is None:
            raise RuntimeError("Run fit() first.")

        T_eff, m, _ = self.phi_t.shape
        if labels is None:
            labels = [f"var_{k+1}" for k in range(m)]

        idx = (
            self.dates[self.prior_length:]
            if self.dates is not None
            else np.arange(T_eff)
        )

        pairs = list(combinations(range(m), 2))
        n_plots = len(pairs)
        rows, cols = self._get_plot_dimension(n_plots)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for ax, (i, j) in zip(axes, pairs):
            name_i = labels[i]
            name_j = labels[j]

            ax.axhline(0.0, linestyle="--", linewidth=0.8)

            if which == "j_to_i":
                # FROM j TO i: phi[i, j]
                s = 100.0 * self.phi_t[:, i, j]
                ax.plot(idx, s)
                ax.fill_between(idx, 0, s, alpha=0.35)
                ax.set_title(f"{name_j} → {name_i}", fontsize=14)
            elif which == "i_to_j":
                # FROM i TO j: phi[j, i]
                s = 100.0 * self.phi_t[:, j, i]
                ax.plot(idx, s)
                ax.fill_between(idx, 0, s, alpha=0.35)
                ax.set_title(f"{name_i} → {name_j}", fontsize=14)
            else:  # which == "both"
                # j → i (blue shaded)
                s_j_to_i = 100.0 * self.phi_t[:, i, j]
                ax.plot(idx, s_j_to_i, label=f"{name_j} → {name_i}")
                ax.fill_between(idx, 0, s_j_to_i, alpha=0.35)

                # i → j (red line, no fill)
                s_i_to_j = 100.0 * self.phi_t[:, j, i]
                ax.plot(idx, s_i_to_j, color="#FFC400", label=f"{name_i} → {name_j}", alpha=0.75)

                ax.set_title(f"{name_i} & {name_j}", fontsize=14)
                ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

            ax.grid(False, axis='x')
            ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

        # hide unused axes
        for ax in axes[n_plots:]:
            ax.axis("off")

        if which == "j_to_i":
            suptitle = "Pairwise Directional Connectedness (FROM j → i)"
        elif which == "i_to_j":
            suptitle = "Pairwise Directional Connectedness (FROM i → j)"
        else:
            suptitle = "Pairwise Directional Connectedness (both directions)"

        fig.suptitle(suptitle, fontsize=16)
        fig.tight_layout()
        return fig, axes

    def plot_net_pairwise_subplots(self, labels=None, figsize=(16, 10)):
        """
        Plot net pairwise directional connectedness in subplots.

        npdc[t,i,j] = 100 * (phi[j,i] - phi[i,j])
        > 0 means j is a net transmitter to i, < 0 means i dominates j.

        One subplot per unordered pair (i < j).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from itertools import combinations

        if self.npdc is None:
            raise RuntimeError("Run fit() first.")

        T_eff, m, _ = self.npdc.shape
        if labels is None:
            labels = [f"var_{k+1}" for k in range(m)]

        idx = (self.dates[self.prior_length:]
               if self.dates is not None
               else np.arange(T_eff))

        pairs = list(combinations(range(m), 2))
        n_plots = len(pairs)
        rows, cols = self._get_plot_dimension(n_plots)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()

        for ax, (i, j) in zip(axes, pairs):
            s_ij = self.npdc[:, i, j]  # 100*(phi[j,i] - phi[i,j])

            name_i = labels[i]
            name_j = labels[j]

            ax.axhline(0.0, linestyle="--", linewidth=0.8)
            ax.plot(idx, s_ij)
            ax.fill_between(idx, 0, s_ij, alpha=0.3)

            ax.set_title(f"net {name_i} → {name_j}", fontsize=14)
            ax.grid(False, axis='x')
            ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

        for ax in axes[n_plots:]:
            ax.axis("off")

        fig.suptitle("Net Pairwise Directional Connectedness", fontsize=16)
        fig.tight_layout()
        return fig, axes

    def plot_network(
        self,
        t_idx: int = -1,
        labels=None,
        threshold: float = 1.0,
        use_net: bool = False,
        figsize=(8, 8),
        ax=None,
    ):
        """
        Plot a connectedness network for a given time index.

        If `ax` is None, a new figure is created. If an Axes is passed,
        the network is drawn into that subplot (useful for panels).

        Requires the `networkx` package. If it is not installed, an
        informative ImportError is raised.

        Parameters
        ----------
        t_idx : int
            Time index in the effective sample (0 .. T_eff-1).
            Negative values count from the end, e.g. -1 = last point.
        labels : list of str or None
            Variable names (length = m). If None, uses var_1, ..., var_m.
        threshold : float
            Minimum edge weight (in percent) to plot. Higher values
            prune weaker links.
        use_net : bool
            If False (default): edges are based on directional GFEVD
                j -> i with weight 100 * phi[i,j].
            If True: edges are based on net pairwise connectedness:
                for each pair (i,j), draw a single arrow from the
                net transmitter to the receiver, with width
                proportional to |NPDC_ij|.
        figsize : tuple
            Figure size if ax is None.
        ax : matplotlib Axes or None
            Existing axes to draw into.

        Returns
        -------
        (fig, ax)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as colors

        # check for networkx
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "TVPVARConnectedness.plot_network requires the 'networkx' package. "
                "Please install it with `pip install networkx` and re-run."
            ) from e

        if self.phi_t is None:
            raise RuntimeError("Run fit() first.")

        T_eff = self.phi_t.shape[0]
        # handle negative indices
        if t_idx < 0:
            t_idx = T_eff + t_idx
        if t_idx < 0 or t_idx >= T_eff:
            raise ValueError(f"t_idx must be in [0, {T_eff-1}], got {t_idx}.")

        m = self.m
        phi = self.phi_t[t_idx, :, :]

        if labels is None:
            labels = [f"var_{k+1}" for k in range(m)]

        # directional measures at this time
        to_i = self.to_dir[t_idx, :]      # TO others, percent
        from_i = self.from_dir[t_idx, :]  # FROM others, percent
        net_i = self.net_dir[t_idx, :]    # TO - FROM, percent
        total_i = to_i + from_i           # total connectedness

        # build graph
        G = nx.DiGraph()
        for k, name in enumerate(labels):
            G.add_node(name, net=net_i[k], total=total_i[k])

        # ----- Build edge lists -----
        edges_label = []   # (src_label, tgt_label)
        edges_ij = []      # (src_index, tgt_index)
        weights = []
        edge_colors = []

        if not use_net:
            # gross directional network: j -> i with weight 100 * phi[i,j]
            for i in range(m):
                for j in range(m):
                    if i == j:
                        continue
                    w = 100.0 * phi[i, j]
                    if w < threshold:
                        continue
                    src = labels[j]
                    tgt = labels[i]
                    edges_label.append((src, tgt))
                    edges_ij.append((j, i))  # from j to i
                    weights.append(w)
                    edge_colors.append("0.6")  # grey
        else:
            # net pairwise network: single arrow per pair
            npdc_t = self.npdc[t_idx, :, :]  # already in percent
            for i in range(m):
                for j in range(i + 1, m):
                    val = npdc_t[i, j]
                    if abs(val) < threshold:
                        continue
                    if val > 0:
                        # i is net transmitter to j
                        src_idx, tgt_idx = i, j
                        w = val
                    else:
                        # j is net transmitter to i
                        src_idx, tgt_idx = j, i
                        w = -val
                    src = labels[src_idx]
                    tgt = labels[tgt_idx]
                    edges_label.append((src, tgt))
                    edges_ij.append((src_idx, tgt_idx))
                    weights.append(w)
                    edge_colors.append("brown")

        # layout
        if len(G.nodes) == 0:
            raise RuntimeError("No nodes in graph.")
        pos = nx.spring_layout(G, k=0.7, iterations=200, seed=1)

        # node sizes (scale total_i)
        min_size, max_size = 400, 1600
        if total_i.max() > 0:
            node_sizes = min_size + (max_size - min_size) * (total_i / total_i.max())
        else:
            node_sizes = np.full(m, (min_size + max_size) / 2)

        # node colours based on NET (TO - FROM)
        cmap = cm.get_cmap("RdYlGn")  # green = receiver, red = transmitter
        vmin = net_i.min()
        vmax = net_i.max()
        if vmin == vmax:
            vmin = -abs(vmax) if vmax != 0 else -1
            vmax = abs(vmax) if vmax != 0 else 1
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        node_colors = cmap(norm(net_i))

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            fig = ax.figure

        # draw nodes and labels
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            ax=ax,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=11,
            font_weight="bold",
            ax=ax,
        )

        # ----- Draw edges with big, visible arrows -----
        if edges_label:
            max_w = max(weights)
            widths = [0.6 + 4.4 * (w / max_w) for w in weights]

            if not use_net:
                # split into single-direction and bidirectional pairs
                from collections import defaultdict
                pair_to_idx = defaultdict(list)
                for idx0, (src_idx, tgt_idx) in enumerate(edges_ij):
                    pair = tuple(sorted((src_idx, tgt_idx)))
                    pair_to_idx[pair].append(idx0)

                # prepare three sets: straight, arc+, arc-
                straight_edges, straight_w, straight_c = [], [], []
                arc_pos_edges, arc_pos_w, arc_pos_c = [], [], []
                arc_neg_edges, arc_neg_w, arc_neg_c = [], [], []

                for pair, idx_list in pair_to_idx.items():
                    if len(idx_list) == 1:
                        k = idx_list[0]
                        straight_edges.append(edges_label[k])
                        straight_w.append(widths[k])
                        straight_c.append(edge_colors[k])
                    else:
                        # two directions between same pair
                        k1, k2 = idx_list[:2]
                        arc_pos_edges.append(edges_label[k1])
                        arc_pos_w.append(widths[k1])
                        arc_pos_c.append(edge_colors[k1])
                        arc_neg_edges.append(edges_label[k2])
                        arc_neg_w.append(widths[k2])
                        arc_neg_c.append(edge_colors[k2])

                common_edge_kwargs = dict(
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=24,           # bigger arrowheads
                    min_source_margin=12,   # move arrowhead off node
                    min_target_margin=12,
                    ax=ax,
                )

                if straight_edges:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=straight_edges,
                        width=straight_w,
                        edge_color=straight_c,
                        connectionstyle="arc3,rad=0.0",
                        **common_edge_kwargs,
                    )

                if arc_pos_edges:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=arc_pos_edges,
                        width=arc_pos_w,
                        edge_color=arc_pos_c,
                        connectionstyle="arc3,rad=0.25",
                        **common_edge_kwargs,
                    )

                if arc_neg_edges:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=arc_neg_edges,
                        width=arc_neg_w,
                        edge_color=arc_neg_c,
                        connectionstyle="arc3,rad=-0.25",
                        **common_edge_kwargs,
                    )
            else:
                # net network, at most one edge per pair, so simple straight arrows are fine
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=edges_label,
                    width=widths,
                    edge_color=edge_colors,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=24,
                    min_source_margin=12,
                    min_target_margin=12,
                    connectionstyle="arc3,rad=0.0",
                    ax=ax,
                )

        ax.set_axis_off()

        # title with date and TCI
        if self.dates is not None:
            import pandas as pd
            eff_dates = pd.to_datetime(self.dates[self.prior_length:])
            date_str = eff_dates[t_idx].strftime("%Y-%m-%d")
        else:
            date_str = f"t_idx={t_idx}"
        tci_val = self.tci[t_idx]
        net_type = "Net" if use_net else "Gross"
        ax.set_title(
            f"{net_type} connectedness on {date_str} (Index = {tci_val:.1f}%)",
            fontsize=10,
        )

        if created_fig:
            fig.tight_layout()

        return fig, ax

    def plot_network_panel(
        self,
        times,
        labels=None,
        threshold: float = 1.0,
        use_net: bool = False,
        figsize_per_plot=(5.0, 5.0),
    ):
        """
        Plot up to four connectedness networks (gross or net) as subplots
        for different dates / time indices, similar to Diebold and Yilmaz (2014).

        Parameters
        ----------
        times : list-like
            List of 1–4 elements. Each element can be:
            - int   : time index in the effective sample (0 .. T_eff-1).
                      Negative values count from the end (e.g. -1 = last).
            - date-like (str, datetime) IF self.dates is available:
                      must match an effective sample date exactly.
        labels : list of str or None
            Variable names.
        threshold : float
            Minimum edge weight (percent) to plot.
        use_net : bool
            False  -> gross directional network.
            True   -> net pairwise network.
        figsize_per_plot : (float, float)
            Size of each subplot; total figure size is scaled by rows/cols.

        Returns
        -------
        (fig, axes_used)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # normalise times to a list
        if not hasattr(times, "__iter__") or isinstance(times, (str, bytes)):
            times = [times]
        times = list(times)
        n = len(times)
        if n < 1 or n > 4:
            raise ValueError("`times` must contain between 1 and 4 elements.")

        if self.phi_t is None:
            raise RuntimeError("Run fit() first.")

        T_eff = self.phi_t.shape[0]

        # effective dates (if available)
        eff_dates = None
        if self.dates is not None:
            import pandas as pd
            eff_dates = pd.to_datetime(self.dates[self.prior_length:])

        idx_list = []
        for t in times:
            if isinstance(t, int):
                idx = t
                if idx < 0:
                    idx = T_eff + idx
                if idx < 0 or idx >= T_eff:
                    raise ValueError(f"Index {t} resolves to {idx}, "
                                     f"but must be in [0, {T_eff-1}].")
                idx_list.append(idx)
            else:
                if eff_dates is None:
                    raise ValueError(
                        "Non-integer `times` entries require `self.dates` to be set."
                    )
                import pandas as pd
                target = pd.to_datetime(t)
                matches = np.where(eff_dates == target)[0]
                if len(matches) == 0:
                    raise ValueError(f"Date {t} not found in effective sample.")
                idx_list.append(int(matches[0]))

        # choose grid: 1x1, 1x2, or 2x2
        if n == 1:
            rows, cols = 1, 1
        elif n == 2:
            rows, cols = 1, 2
        else:  # 3 or 4
            rows, cols = 2, 2

        fig_w = cols * figsize_per_plot[0]
        fig_h = rows * figsize_per_plot[1]
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.atleast_1d(axes).flatten()

        for ax, idx in zip(axes, idx_list):
            # draw into given axis
            self.plot_network(
                t_idx=idx,
                labels=labels,
                threshold=threshold,
                use_net=use_net,
                figsize=figsize_per_plot,
                ax=ax,
            )

        # turn off unused axes
        for ax in axes[len(idx_list):]:
            ax.axis("off")

        fig.tight_layout()
        return fig, axes[: len(idx_list)]
