from raster_vector.projection import compute_utm_from_lat_lon


def test_utm_northern_hemisphere():
    """Test UTM calculation for northern hemisphere (e.g., Paris)."""
    # Paris: 48.8566 N, 2.3522 E -> Zone 31N -> EPSG:32631
    epsg = compute_utm_from_lat_lon(48.8566, 2.3522)
    assert epsg == 32631


def test_utm_southern_hemisphere():
    """Test UTM calculation for southern hemisphere (e.g., Sydney)."""
    # Sydney: 33.8688 S, 151.2093 E -> Zone 56S -> EPSG:32756
    epsg = compute_utm_from_lat_lon(-33.8688, 151.2093)
    assert epsg == 32756


def test_utm_near_antimeridian():
    """Test UTM calculation near the antimeridian."""
    # Near Suva, Fiji: 18.1416 S, 178.4419 E -> Zone 60S -> EPSG:32760
    epsg = compute_utm_from_lat_lon(-18.1416, 178.4419)
    assert epsg == 32760
