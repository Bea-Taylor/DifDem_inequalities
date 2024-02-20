import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from geopy.geocoders import Nominatim
from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad

def compute_health_domain(params):
    population_data_path = params['population_data_path']
    QOF_data_path = params['QOF_data_path']
    GP_catchment_areas_geo_path = params['GP_catchment_areas_geo_path']
    LSOA_geo_path = params['LSOA_geo_path']
    LSOA_GP_intersections_path = params['LSOA_GP_intersections_path']

    N_intersections = params['N_intersections']

    df_pop = pd.read_csv(population_data_path)
    df_QOF = pd.read_csv(QOF_data_path)
    gdf_catchment = gpd.read_file(GP_catchment_areas_geo_path)
    gdf_LSOA = gpd.read_file(LSOA_geo_path)
    LSOA_GP_intersections = np.load(LSOA_GP_intersections_path, allow_pickle=True)

    QOF_domains = list(df_QOF.columns[3:])

    # Change to lon lat
    gdf_LSOA = gdf_LSOA.to_crs(epsg=4326)
    # Remove Wales
    gdf_LSOA = gdf_LSOA.drop(gdf_LSOA[gdf_LSOA['LSOA11CD'].str.startswith('W')].index)
    # Add population
    gdf_LSOA = gdf_LSOA.merge(df_pop, left_on='LSOA11CD', right_on='LSOA11CD')

    # Replace missing data in QOF data with 0
    for d in QOF_domains:
        if df_QOF[df_QOF[d] == '-'].shape[0] > 0:
            df_QOF.loc[df_QOF[d] == '-', d] = 0
        if df_QOF[df_QOF[d] == 'Insufficient indicator data'].shape[0] > 0:
            df_QOF.loc[df_QOF[d] == 'Insufficient indicator data', d] = 0
            
    # Remove commas in QOF data
    for d in QOF_domains:
        df_QOF.loc[:, d] = df_QOF.loc[:, d].astype(str).str.replace(',', '').astype(float)
    df_QOF.loc[:, 'list_pop'] = df_QOF.loc[:, 'list_pop'].astype(str).str.replace(',', '').astype(float)

    # Merge with QOF data to ensure that we don't use GPs that have no QOF data
    unmerged_gdf_catchment = gdf_catchment
    gdf_catchment = unmerged_gdf_catchment.merge(df_QOF, left_on='ODSCode', right_on='pcode')

    # Create mercator versions of the geo data
    merc_gdf_catchment = gdf_catchment.to_crs(epsg=3857)
    merc_gdf_LSOA = gdf_LSOA.to_crs(epsg=3857)

    # Compute the population $I_{ik}$ of the intersection of LSOA $i$ and GP $k$

    j = 0
    for i, LSOA_row in gdf_LSOA.iterrows():
        intersection_data = LSOA_GP_intersections[j]; j += 1
        LSOA_area = polygon_area(LSOA_row['geometry'])

        for k in range(len(intersection_data)):
            GP_i = intersection_data[k][0]
            int_area = intersection_data[k][1]
            int_pop = (int_area / LSOA_area) * LSOA_row['pop']
            intersection_data[k].append(int_pop)

    ### Compute $\sum_j I_{jk}$, used for computing $Z_i^\text{LSOA}$ and $L_i^\text{LSOA}$.

    GPs_covered_area = np.full(np.max(gdf_catchment.index+1), -1, dtype=float)

    print(len(GPs_covered_area))

    for intersection_data in LSOA_GP_intersections:
        for GP_i, GP_intersection_area, GP_intersection_num_patients in intersection_data:
            if GPs_covered_area[GP_i] == -1:
                GPs_covered_area[GP_i] = 0
            GPs_covered_area[GP_i] += GP_intersection_area


    # Store the patient and afflicted numbers, to speed up the lookup

    GPs_list_pop = np.full(np.max(gdf_catchment.index+1), -1, dtype=float)
    GPs_afflicted = {}
    for QOF_domain_key in QOF_domains:
        GPs_afflicted[QOF_domain_key] = np.full(np.max(gdf_catchment.index+1), -1, dtype=float)

    for i, row in gdf_catchment.iterrows():
        GPs_list_pop[i] = row['list_pop']
        for QOF_domain_key in QOF_domains:
            GPs_afflicted[QOF_domain_key][i] = row[QOF_domain_key]

    # Estimate LSOA prevalence rates

    print('Estimating LSOA prevalence rates')

    LSOA_prevalence_rates = {}

    for QOF_domain_key in QOF_domains:
        LSOA_prevalence_rates[QOF_domain_key] = []

    l = 0
    for intersection_data in LSOA_GP_intersections:
        _list_pop = 0
        for GP_i, GP_intersection_area, GP_intersection_num_patients in intersection_data:
            _list_pop += (GP_intersection_area/GPs_covered_area[GP_i]) * GPs_list_pop[GP_i]

            if GPs_list_pop[GP_i] == -1:
                raise Exception('Invalid GP used.')

        for QOF_domain_key in QOF_domains:
            _afflicted = 0
            for GP_i, GP_intersection_area, GP_intersection_num_patients in intersection_data:
                _afflicted += (GP_intersection_area/GPs_covered_area[GP_i]) * GPs_afflicted[QOF_domain_key][GP_i]

                if GPs_afflicted[QOF_domain_key][GP_i] == -1:
                    raise Exception('Invalid GP used.')

            if _list_pop == 0:
                LSOA_prevalence_rates[QOF_domain_key].append(-1)
            else:
                LSOA_prevalence_rates[QOF_domain_key].append(_afflicted/_list_pop)

    for QOF_domain_key in QOF_domains:
        gdf_LSOA['%s_prevalence_rate' % QOF_domain_key] = LSOA_prevalence_rates[QOF_domain_key]

    # For LSOAs that do not coincide with any GPs, estimate their prevalence rates by taking the simple averages of their neighbours

    print('Estimate prevalence rates of LSOAs outside of GP catchment areas.')

    QOF_domain_key0 = QOF_domains[0]
    LSOAs_with_no_GPs = gdf_LSOA.index[gdf_LSOA['%s_prevalence_rate' % QOF_domain_key0] == -1]

    while len(LSOAs_with_no_GPs) != 0:

        for j, LSOA_i in enumerate(LSOAs_with_no_GPs):
            print('%s/%s' % (j+1, len(LSOAs_with_no_GPs)), end='\r')
            LSOA_row = merc_gdf_LSOA.loc[LSOA_i]

            neighbours = set(merc_gdf_LSOA.index[merc_gdf_LSOA['geometry'].touches(LSOA_row['geometry'])])
            neighbours = neighbours.union(set(merc_gdf_LSOA.index[merc_gdf_LSOA['geometry'].overlaps(LSOA_row['geometry'])]))
            neighbours = list(neighbours)

            for QOF_domain_key in QOF_domains:
                avg_prevalence_rate = 0
                avg_prevalence_rate_n = 0
                for neighbour_LSOA_i in neighbours:
                    if gdf_LSOA.loc[neighbour_LSOA_i]['%s_prevalence_rate' % QOF_domain_key] != -1:
                        avg_prevalence_rate += gdf_LSOA.loc[neighbour_LSOA_i]['%s_prevalence_rate' % QOF_domain_key]
                        avg_prevalence_rate_n += 1

                if avg_prevalence_rate_n != 0:
                    avg_prevalence_rate = avg_prevalence_rate / avg_prevalence_rate_n
                    gdf_LSOA.loc[LSOA_i, '%s_prevalence_rate' % QOF_domain_key] = avg_prevalence_rate

        LSOAs_with_no_GPs = gdf_LSOA.index[gdf_LSOA['%s_prevalence_rate' % QOF_domain_key0] == -1]

    # Compute the estimated number of afflicted

    for QOF_domain_key in QOF_domains:
        gdf_LSOA['%s_afflicted' % QOF_domain_key] = gdf_LSOA['%s_prevalence_rate' % QOF_domain_key] * gdf_LSOA['pop']

    df_health_domain = gdf_LSOA.loc[:, ['LSOA11CD', 'LSOA11NM'] + ['%s_prevalence_rate' % d for d in QOF_domains] + ['%s_afflicted' % d for d in QOF_domains]]
    
    return df_health_domain


def compute_LSOA_and_GP_intersections(params):
    population_data_path = params['population_data_path']
    QOF_data_path = params['QOF_data_path']
    GP_catchment_areas_geo_path = params['GP_catchment_areas_geo_path']
    LSOA_geo_path = params['LSOA_geo_path']
    
    N_intersections = params['N_intersections']
    
    df_pop = pd.read_csv(population_data_path)
    df_QOF = pd.read_csv(QOF_data_path)
    gdf_catchment = gpd.read_file(GP_catchment_areas_geo_path)
    gdf_LSOA = gpd.read_file(LSOA_geo_path)
    
    # Change to lon lat
    gdf_LSOA = gdf_LSOA.to_crs(epsg=4326)
    # Remove Wales
    gdf_LSOA = gdf_LSOA.drop(gdf_LSOA[gdf_LSOA['LSOA11CD'].str.startswith('W')].index)
    # Add population
    gdf_LSOA = gdf_LSOA.merge(df_pop, left_on='LSOA11CD', right_on='LSOA11CD')
    
    # Merge with QOF data to ensure that we don't use GPs that have no QOF data
    unmerged_gdf_catchment = gdf_catchment
    gdf_catchment = unmerged_gdf_catchment.merge(df_QOF, left_on='ODSCode', right_on='pcode')

    # Create mercator versions of the geo data
    merc_gdf_catchment = gdf_catchment.to_crs(epsg=3857)
    merc_gdf_LSOA = gdf_LSOA.to_crs(epsg=3857)
    
    # Compute intersections between LSOAs and GP catchment areas

    LSOA_GP_catchment_area_intersections = []

    GP_xs = np.array(merc_gdf_catchment['geometry'].centroid.x)
    GP_ys = np.array(merc_gdf_catchment['geometry'].centroid.y)

    print('Computing intersections.\n')
    for i, LSOA_row in merc_gdf_LSOA.iterrows():
        print('LSOA %s/%s' % (i+1, merc_gdf_LSOA.shape[0]), end='\r')

        LSOA_center = LSOA_row.geometry.centroid
        dx = GP_xs - LSOA_center.x
        dy = GP_ys - LSOA_center.y
        dists = np.sqrt(dx**2 + dy**2)
        closest_GPs = merc_gdf_catchment.index[dists.argsort()[:N_intersections]]

        inters = []

        for j in closest_GPs:
            inter = merc_gdf_catchment.loc[j].geometry.buffer(0).intersection(LSOA_row.geometry)
            inters.append(inter)

        gdf_inters = gpd.GeoDataFrame({ 'geometry' : inters })
        gdf_inters.crs = {'init': 'epsg:3857', 'no_defs': True}
        gdf_inters = gdf_inters.to_crs(epsg=4326)

        _intersection_areas = []

        for k, j in enumerate(closest_GPs):
            inter = gdf_inters.iloc[k]['geometry']
            if inter.area != 0:
                _intersection_areas.append([j, polygon_area(inter)])

        LSOA_GP_catchment_area_intersections.append(_intersection_areas)
    
    LSOA_GP_catchment_area_intersections = np.array(LSOA_GP_catchment_area_intersections, dtype=object)
    return LSOA_GP_catchment_area_intersections
    
def polygon_area(geom, radius = 6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """
    
    # If geom is MultiPolygon, loop over the individual polygons
    if type(geom) is shapely.geometry.MultiPolygon:
        total_area = 0
        for sub_geom in geom.geoms:
            total_area += polygon_area(sub_geom)
        return total_area
    
    # If geom has interior, then subtract it
    if hasattr(geom, 'interiors'):
        if len(geom.interiors) > 0:
            area = polygon_area(shapely.geometry.Polygon(geom.exterior))
            for interior in geom.interiors:
                area -= polygon_area(shapely.geometry.Polygon(interior))
            return area

    geom = np.array(geom.boundary.coords)
    lats = geom[:,1]
    lons = geom[:,0]
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)

    # Line integral based on Green's Theorem, assumes spherical Earth

    #close polygon
    if lats[0]!=lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    #colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
    colat = 2*arctan2( sqrt(a), sqrt(1-a) )

    #azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas=diff(colat)/2
    colat=colat[0:-1]+deltas

    # Perform integral
    integrands = (1-cos(colat)) * daz

    # Integrate 
    area = abs(sum(integrands))/(4*pi)

    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return area * 4*pi*radius**2
    else: #return in ratio of sphere total area
        return area