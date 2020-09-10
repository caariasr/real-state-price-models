from scipy.spatial import KDTree
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd


def add_extra_features(data):
    # add the valid period in months
    numeric_vars = [
        'offering_type_id', 'bedroom_id', 'property_sqft',
        'property_cheques', 'coordinates_lat', 'coordinates_lon'
    ]
    data[numeric_vars] = data[numeric_vars].astype('float')
    data['meta_valid_from_dts'] = pd.to_datetime(data['meta_valid_from_dts'])
    data['meta_valid_to_dts'] = pd.to_datetime(data['meta_valid_to_dts'])
    data['property_log_sqft'] = np.log(data['property_sqft'])
    data['valid_period'] = (
            data['meta_valid_to_dts'] -
            data['meta_valid_from_dts']
    )
    data['valid_period'] = data['valid_period'].apply(
        lambda x: round(x.total_seconds() / (60 * 60 * 24 * 30.4167), 2))
    # time stamp groups for start date of listing being valid
    cond_list = [
        data.meta_valid_from_dts < '2018-10-27',
        ((data.meta_valid_from_dts >= '2018-10-27') &
         (data.meta_valid_from_dts < '2018-10-28')),
        ((data.meta_valid_from_dts >= '2018-10-28') &
         (data.meta_valid_from_dts < '2018-10-29')),
        ((data.meta_valid_from_dts >= '2018-10-29') &
         (data.meta_valid_from_dts < '2018-10-30')),
        ((data.meta_valid_from_dts >= '2018-10-30') &
         (data.meta_valid_from_dts < '2018-10-31')),
        ((data.meta_valid_from_dts >= '2018-10-31') &
         (data.meta_valid_from_dts < '2018-11-01'))
    ]
    choice_list = [
        'before_oct_27th_2018',
        'oct_27th_2018',
        'oct_28th_2018',
        'oct_29th_2018',
        'oct_30th_2018',
        'oct_31th_2018'
    ]
    data['ts_groups_from'] = np.select(
        cond_list, choice_list)
    new_vars = pd.get_dummies(data.ts_groups_from)
    choice = list(new_vars.columns)[0]
    for grp in choice_list:
        if grp != choice:
            new_vars[grp] = 0
    data = data.join(new_vars)
    # distance features
    poi_df = pd.read_csv("static/csv/poi.csv")
    data['metro_poi'] = nearest_poi(
        data, poi_df, 'Metro')
    data['tram_poi'] = nearest_poi(
        data, poi_df, 'Tram')
    new_att = add_landmarks(data, poi_df)
    data = data.join(new_att)
    covariates = [
        'offering_type_id',
        'bedroom_id',
        'bathroom_id',
        'property_log_sqft',
        'property_cheques',
        'valid_period'
    ] + choice_list + [
        'metro_poi',
        'tram_poi'
    ] + list(new_att.columns)
    return data[covariates]


def add_landmarks(df, poi):
    landmark_df = poi.loc[poi.category == 'Landmark'].reset_index()
    all_dists = distance_matrix(
        df[
            ['coordinates_lat',
             'coordinates_lon']
        ].values,
        landmark_df[['lat', 'lon']].values
    )
    dists = pd.DataFrame(all_dists)
    dists.columns = [
        x.lower().replace(' ', '_')
        for x in list(landmark_df.name)
    ]
    return dists


def nearest_poi(listings, poi, poi_type):
    cat_df = poi.loc[poi.category == poi_type].reset_index()
    tree = KDTree(
        cat_df[['lat', 'lon']].values
    )
    neighbor_dists, neighbor_indices = tree.query(
        listings[
            ['coordinates_lat', 'coordinates_lon']
        ].values
    )
    return neighbor_dists