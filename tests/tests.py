import pytest

import numpy as np
import matplotlib.pyplot as plt
import mopc as mop
from mopc.beam import f_beam_150

from mopc.config import mopc_datadir
from mopc.gnfw import GeneralizedNFWProfile


"""choose halo redshift and mass [Msun]
"""
z = 0.57
m = 2e13

"""choose radial range (x = r/rvir)
"""
x = np.logspace(np.log10(0.1), np.log10(10), 100)

"""choose angular range [eg. arcmin]
"""
theta = np.arange(100) * 0.05 + 0.5
sr2sqarcmin = 3282.8 * 60.0 ** 2

"""
choose observing frequency [GHz]
"""
nu = 150.0


@pytest.mark.parametrize('use_mass_distribution', [True, False])
def test_density(use_mass_distribution, write_data=True):
    """Test density projection
    """
    rho0 = np.log10(4e3 * (m / 1e14) ** 0.29 * (1 + z) ** (-0.66))
    xc = 0.5
    bt = 3.83 * (m / 1e14) ** 0.04 * (1 + z) ** (-0.025)

    par_rho = [rho0, xc, bt, 1]

    if use_mass_distribution:
        masses = np.loadtxt(mopc_datadir.joinpath("mass_distrib.txt"))
        gnfw = GeneralizedNFWProfile(nbins=10, masses=masses)
    else:
        gnfw = GeneralizedNFWProfile(nbins=10)

    rho_gnfw = gnfw.rho(par_rho)

    temp_ksz_gnfw = mop.make_a_obs_profile_sim_rho(theta, m, z, par_rho, f_beam_150)

    assert np.isfinite(temp_ksz_gnfw).all()

    if use_mass_distribution:
        data_path = mopc_datadir.joinpath("test_density_mass_distrib.npz")
    else:
        data_path = mopc_datadir.joinpath("test_density.npz")

    if not data_path.is_file():
        write_data = True

    if write_data:
        np.savez(data_path, rho=rho_gnfw, temp_ksz=temp_ksz_gnfw)

    data = np.load(data_path)

    assert np.allclose(rho_gnfw, data["rho"])
    assert np.allclose(temp_ksz_gnfw, data["temp_ksz"])


@pytest.mark.parametrize('use_mass_distribution', [True, False])
def test_pressure(use_mass_distribution, write_data=True):
    """Test pressure projection
    """
    P0 = 18.1 * (m / 1e14) ** 0.154 * (1 + z) ** (-0.758)
    al = 1.0
    bt = 4.35 * (m / 1e14) ** 0.0393 * (1 + z) ** 0.415

    if use_mass_distribution:
        masses = np.loadtxt(mopc_datadir.joinpath("mass_distrib.txt"))
        gnfw = GeneralizedNFWProfile(nbins=10, masses=masses)
    else:
        gnfw = GeneralizedNFWProfile(nbins=10)

    par_pth = [P0, al, bt, 1]
    pth_gnfw = gnfw.thermal_pressure(par_pth)

    # pth_gnfw = mop.Pth_gnfw1h(x, m, z, par_pth)
    temp_tsz_gnfw = mop.make_a_obs_profile_sim_pth(theta, m, z, par_pth, nu, f_beam_150)

    assert np.isfinite(temp_tsz_gnfw).all()

    if use_mass_distribution:
        data_path = mopc_datadir.joinpath("test_pressure_mass_distrib.npz")
    else:
        data_path = mopc_datadir.joinpath("test_pressure.npz")

    if not data_path.is_file():
        write_data = True

    if write_data:
        np.savez(data_path, pressure=pth_gnfw, temp_tsz=temp_tsz_gnfw)

    data = np.load(data_path)

    assert np.allclose(pth_gnfw, data["pressure"])
    assert np.allclose(temp_tsz_gnfw, data["temp_tsz"])
