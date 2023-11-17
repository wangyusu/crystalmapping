from crystalmapping import peakindexer
from crystalmapping.ubmatrix import UBMatrix, _get_u_from_geo, _get_rot_matrix, _get_U_from_cart_and_inst
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pyFAI
from dataclasses import dataclass
import typing

from crystalmapping.peakindexer import PeakIndexer, IndexerConfig
import xarray as xr
import numpy as np


@dataclass
class IndexByU(PeakIndexer):

    def index_with_U(self, U) -> None:
        """indexing new data with previously found U matrix

        parameters
        ----------
        numU: int
            the index of the U matrix want to use in the candidate list
        """
        V1 = [_get_u_from_geo(row.x, row.y, self._ai).T for row in self._peaks.itertuples()]
        Rs = [_get_rot_matrix(row.alpha, row.beta, row.gamma) for row in self._peaks.itertuples()]
        invB = self._ubmatrix.invB

        hkls = []
        for v1, R in zip(V1, Rs):
            v4 = invB @ U.T @ R.T @ v1
            hkls.append(v4.T)

        losses = peakindexer._get_losses(self._ubmatrix, np.array(hkls))
        loss = np.min(losses)

        ac = peakindexer.AngleComparsion(
            None,
            None,
            50.04, #self._previous_result["angle_sample"][numU].item(),
            49.96, #self._previous_result["angle_grain"][numU].item(),
            0.09   #self._previous_result["diff_angle"][numU].item(),
        )

        ir = peakindexer.IndexResult(
            np.array('13_1', dtype=object), #self._previous_result["peak1"][numU].data,
            np.array('12_1', dtype=object), #self._previous_result["peak2"][numU].data,
            np.array([[-0.982, 0.186, -0.029], [-0.084, -0.292, 0.953], [0.168, 0.938, 0.302]]), #self._previous_result.U[numU].data,
            hkls,
            losses,
            loss,
            ac
        )

        self._peak_index = peakindexer._make_peak_index([ir], self._peaks.index)
        return

    def get_data(self):
        return self._peak_index.sel({"candidate": 0}).sortby("losses")
