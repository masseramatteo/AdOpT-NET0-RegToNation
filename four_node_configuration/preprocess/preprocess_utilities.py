import numpy as np
from math import radians, cos, sin


def euclidean_distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def generate_line_cluster(N, D):
    """
    Genera N punti su una linea orizzontale centrata in (0,0), distanziati di D.
    Per N=2: [(-D/2, 0), (+D/2, 0)]
    """
    if N < 1:
        return []

    # posizioni equispaziate lungo x, centrate
    xs = (np.arange(N) - (N - 1) / 2.0) * D
    ys = np.zeros(N)
    return list(zip(xs.astype(float), ys.astype(float)))


def rotate(points, theta_deg):
    """
    Ruota una lista di punti (x,y) di theta_deg intorno all'origine.
    """
    th = radians(theta_deg)
    c, s = cos(th), sin(th)
    rot = []
    for x, y in points:
        xr = c * x - s * y
        yr = s * x + c * y
        rot.append((float(xr), float(yr)))
    return rot


def shift(points, dx, dy):
    """
    Trasla una lista di punti (x,y) di (dx,dy).
    """
    return [(float(x + dx), float(y + dy)) for x, y in points]


def polar_to_cartesian(r, theta_deg):
    """
    Converte coordinate polari (r, theta_deg) in cartesiane (x,y).
    """
    th = radians(theta_deg)
    return (float(r * cos(th)), float(r * sin(th)))


def cartesian_to_lonlat(x_km, y_km, lon0, lat0, km_per_deg_lon, km_per_deg_lat):
    """
    Conversione approssimata km -> lon/lat:
      lon = lon0 + x_km / km_per_deg_lon
      lat = lat0 + y_km / km_per_deg_lat
    """
    lon = lon0 + (x_km / km_per_deg_lon)
    lat = lat0 + (y_km / km_per_deg_lat)
    return float(lon), float(lat)

def generate_square_cluster(N, D):
    """
    Genera N punti su una griglia quadrata centrata in (0,0) con passo D.
    Per N=2 restituisce due punti vicini sulla griglia.
    """
    if N < 1:
        return []
    side = int(np.ceil(np.sqrt(N)))
    coords = []
    start = -(side - 1) / 2.0 * D
    for i in range(side):
        for j in range(side):
            coords.append((start + i * D, start + j * D))
    coords = coords[:N]
    return [(float(x), float(y)) for x, y in coords]

def latin_hypercube(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """
    LHS classico: restituisce un array (n,d) in [0,1).
    """
    rng = np.random.default_rng(seed)
    H = np.empty((n, d), dtype=float)

    # per ogni dimensione: n intervalli, un campione per intervallo, poi permutazione
    for j in range(d):
        cut = np.linspace(0, 1, n + 1)
        u = rng.random(n)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])  # un punto per strato
        rng.shuffle(points)
        H[:, j] = points

    return H


def map_to_nearest(values, x):
    """
    Mappa un valore continuo x al valore più vicino in una lista discreta 'values'.
    """
    values = np.asarray(values, dtype=float)
    idx = int(np.argmin(np.abs(values - x)))
    return float(values[idx])


def map_to_discrete_index(values, u):
    """
    Mappa u in [0,1) a un indice discreto uniforme.
    """
    n = len(values)
    idx = int(np.floor(u * n))
    if idx == n:
        idx = n - 1
    return values[idx]
