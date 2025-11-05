

def compute_utm_from_lat_lon(lat, lon):
    epsg = int(
        32700 - 
        round(( 45 + lat) / 90, 0) * 100 + 
        round((183 + lon) /  6, 0)
    )
    return epsg
