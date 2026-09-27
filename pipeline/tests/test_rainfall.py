import numpy as np
import pandas as pd
import pytest

from bits_ml import rainfall as rf

N = rf.NLAT * rf.NLON
LATS = rf.LAT0 + rf.STEP * np.arange(rf.NLAT)
LONS = rf.LON0 + rf.STEP * np.arange(rf.NLON)
IMD_2020 = rf.IMD_DIR / "imd_rf25_2020.nc"
HAVE_DATA = IMD_2020.exists() and rf.CGWB_CSV.exists()


def field(fn):
    la, lo = np.meshgrid(LATS, LONS, indexing="ij")
    return fn(la, lo).reshape(-1)


def random_points(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(LATS[0], LATS[-1], n), rng.uniform(LONS[0], LONS[-1], n)


def test_linear_field_is_reproduced_exactly_at_arbitrary_points():
    lat, lon = random_points()
    got = rf.sample(field(lambda a, b: 3.0 * a - 2.0 * b + 300.0), rf.bilinear_stencil(lat, lon))
    np.testing.assert_allclose(got, 3.0 * lat - 2.0 * lon + 300.0, rtol=0, atol=1e-9)


def test_bilinear_surface_with_cross_term_is_reproduced_exactly():
    lat, lon = random_points(seed=1)
    got = rf.sample(field(lambda a, b: 5.0 + a * b), rf.bilinear_stencil(lat, lon))
    np.testing.assert_allclose(got, 5.0 + lat * lon, rtol=0, atol=1e-9)


def test_node_coordinates_return_the_node_value_including_grid_edges():
    values = np.arange(N, dtype=float)
    i = np.array([0, 40, rf.NLAT - 1, rf.NLAT - 1, 0])
    j = np.array([0, 77, rf.NLON - 1, 0, rf.NLON - 1])
    got = rf.sample(values, rf.bilinear_stencil(LATS[i], LONS[j]))
    np.testing.assert_allclose(got, i * rf.NLON + j, rtol=0, atol=1e-9)


def test_weights_are_non_negative_and_sum_to_one():
    st = rf.bilinear_stencil(*random_points(seed=2))
    assert (st.w >= 0).all()
    np.testing.assert_allclose(st.w.sum(axis=1), 1.0, atol=1e-12)


def test_sea_node_is_dropped_and_remaining_weights_renormalised():
    st = rf.bilinear_stencil(np.array([20.1]), np.array([80.2]))
    grid = np.full(N, 7.0)
    a, b, c, d = st.idx[0]
    grid[a], grid[b], grid[c], grid[d] = np.nan, 40.0, 10.0, 10.0
    w = st.w[0]
    expected = (40.0 * w[1] + 10.0 * w[2] + 10.0 * w[3]) / (w[1] + w[2] + w[3])
    assert rf.sample(grid, st)[0] == pytest.approx(expected)


def test_stencil_entirely_at_sea_gives_nan_not_zero():
    st = rf.bilinear_stencil(np.array([20.1]), np.array([80.2]))
    grid = np.full(N, 5.0)
    grid[st.idx[0]] = np.nan
    assert np.isnan(rf.sample(grid, st)[0])


def test_sampling_works_on_a_stack_of_days():
    st = rf.bilinear_stencil(*random_points(n=20, seed=3))
    days = np.stack([np.full(N, float(k)) for k in range(5)])
    got = rf.sample(days, st)
    assert got.shape == (5, 20)
    np.testing.assert_allclose(got, np.arange(5.0)[:, None] * np.ones((1, 20)))


@pytest.mark.parametrize("lat,lon", [(6.4, 80.0), (38.6, 80.0), (20.0, 66.4), (20.0, 100.1)])
def test_points_outside_the_grid_are_rejected(lat, lon):
    with pytest.raises(ValueError):
        rf.bilinear_stencil(np.array([lat]), np.array([lon]))


def test_nearest_stencil_points_at_the_week1_snapped_node():
    lat, lon = random_points(seed=4)
    idx = rf.nearest_stencil(lat, lon).idx[:, 0]
    snap_lat = np.round((lat - 6.5) / 0.25) * 0.25 + 6.5
    snap_lon = np.round((lon - 66.5) / 0.25) * 0.25 + 66.5
    np.testing.assert_allclose(LATS[idx // rf.NLON], snap_lat)
    np.testing.assert_allclose(LONS[idx % rf.NLON], snap_lon)


def test_month_without_any_valid_day_stays_nan():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    daily = np.ones((len(dates), 2), dtype=np.float32)
    daily[dates.month == 3, 1] = np.nan
    rain, wet = rf.monthly_totals(dates, daily)
    assert rain[2, 0] == 31.0
    assert rain[1, 0] == 29.0  # 2020 is a leap year
    assert np.isnan(rain[2, 1]) and np.isnan(wet[2, 1])


def test_wet_day_threshold_is_strictly_above_2_5_mm():
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    daily = np.zeros((len(dates), 1), dtype=np.float32)
    jan = np.flatnonzero(dates.month == 1)
    daily[jan[0], 0], daily[jan[1], 0], daily[jan[2], 0] = 2.5, 2.6, 100.0
    rain, wet = rf.monthly_totals(dates, daily)
    assert wet[0, 0] == 2
    assert rain[0, 0] == pytest.approx(105.1, abs=1e-4)


def test_wettest_day_of_a_month_without_valid_days_stays_nan():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    daily = np.ones((len(dates), 2))
    daily[dates.month == 3, 1] = np.nan
    daily[40, 0] = 9.0  # 10 February
    wettest = rf.monthly_max(dates, daily)
    assert wettest[1, 0] == 9.0 and wettest[2, 0] == 1.0 and np.isnan(wettest[2, 1])


def test_wet_days_and_wettest_day_are_taken_at_nodes_then_interpolated():
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    grid = np.zeros((len(dates), N))
    st = rf.bilinear_stencil(np.array([20.125]), np.array([80.0]))  # halfway between two latitude nodes
    south, _, north, _ = st.idx[0]
    grid[:, south] = 3.0  # wet every day; the north node never is
    grid[10, south] = 80.0
    rain, wet, wettest = rf.well_monthly(dates, grid, st)
    assert rain[0, 0] == pytest.approx((3.0 * 30 + 80.0) / 2)
    assert wet[0, 0] == pytest.approx(15.5)
    assert wettest[0, 0] == pytest.approx(40.0)
    # interpolating first leaves 1.5 mm on most days, below the wet-day threshold
    assert rf.monthly_totals(dates, rf.sample(grid, st))[1][0, 0] == 1


def test_bilinear_is_exact_on_a_finer_grid_whose_step_is_not_exact_in_binary():
    g = rf.Grid(6.525, 66.525, 0.05, 640, 670)
    la, lo = np.meshgrid(g.lats, g.lons, indexing="ij")
    lat = np.array([g.lats[0], 20.0123, g.lats[-1]])
    lon = np.array([g.lons[0], 80.0456, g.lons[-1]])
    got = rf.sample((2.0 * la - lo).reshape(-1), rf.bilinear_stencil(lat, lon, g))
    np.testing.assert_allclose(got, 2.0 * lat - lon, atol=1e-9)


def test_box_stencil_equals_the_mean_of_bilinear_samples_over_the_box():
    lat, lon = random_points(n=300, seed=5)
    lat, lon = np.clip(lat, 7.0, 38.0), np.clip(lon, 67.0, 99.5)
    h = np.random.default_rng(6).choice([0.05, 0.025, 1 / 120], lat.size)
    values = np.random.default_rng(7).gamma(2.0, 20.0, N)
    offsets = (np.arange(5) + 0.5) / 5 * 2 - 1
    subs = [rf.sample(values, rf.bilinear_stencil(lat + h * a, lon + h * b)) for a in offsets for b in offsets]
    np.testing.assert_allclose(rf.sample(values, rf.box_stencil(lat, lon, h)), np.mean(subs, axis=0), rtol=1e-12)


def test_zero_half_width_reduces_to_bilinear_even_next_to_the_sea():
    lat, lon = random_points(n=300, seed=8)
    values = np.random.default_rng(9).gamma(2.0, 20.0, N)
    values[np.random.default_rng(10).random(N) < 0.3] = np.nan
    box, bil = rf.box_stencil(lat, lon, 0.0), rf.bilinear_stencil(lat, lon)
    np.testing.assert_allclose(box.w.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(rf.sample(values, box), rf.sample(values, bil), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (14.1, 78.2, 0.05),
        (14.15, 78.25, 0.025),
        (14.2167, 78.1167, 1 / 120),  # 14 deg 13 min, 78 deg 7 min
        (14.19, 78.16, 0.005),
        (14.1917, 78.1639, 0.0),
    ],
)
def test_coordinate_half_width_follows_the_rounding_lattice(lat, lon, expected):
    assert rf.coordinate_half_width(np.array([lat]), np.array([lon]))[0] == pytest.approx(expected)


@pytest.mark.skipif(not HAVE_DATA, reason="IMD / CGWB data not present")
def test_real_grid_layout_and_land_mask():
    dates, grid = rf.read_year(IMD_2020)
    assert len(dates) == 366
    assert (~np.isnan(grid)).sum(axis=1).min() == (~np.isnan(grid)).sum(axis=1).max() == 4964


@pytest.mark.skipif(not HAVE_DATA, reason="IMD / CGWB data not present")
def test_rain_via_nodes_equals_summing_interpolated_days_on_real_data():
    wells = rf.load_wells()
    lat, lon = wells["Latitude"].to_numpy(), wells["Longitude"].to_numpy()
    dates, grid = rf.read_year(IMD_2020)
    st = rf.box_stencil(lat, lon, rf.coordinate_half_width(lat, lon))
    rain, _, _ = rf.well_monthly(dates, grid, st)
    np.testing.assert_allclose(rain, rf.monthly_totals(dates, rf.sample(grid, st))[0], rtol=1e-9)


@pytest.mark.skipif(not HAVE_DATA, reason="IMD / CGWB data not present")
def test_every_well_gets_rain_including_the_sixteen_sea_snapped():
    wells = rf.load_wells()
    lat, lon = wells["Latitude"].to_numpy(), wells["Longitude"].to_numpy()
    _, grid = rf.read_year(IMD_2020)
    bilinear = rf.sample(grid, rf.bilinear_stencil(lat, lon))
    nearest = rf.sample(grid, rf.nearest_stencil(lat, lon))
    assert not np.isnan(bilinear).any()
    sea_snapped = np.isnan(nearest).all(axis=0)
    assert sea_snapped.sum() == 16
    assert (bilinear[:, sea_snapped].sum(axis=0) > 500).all()
