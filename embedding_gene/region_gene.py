import pandas as pd

data = pd.read_csv('select_poi_data.csv')
data = data.dropna()


# 122w行
def classify(s):
    if (('railway' in s) or ('highway' in s) or ('noexit' in s) or ('traffic' in s) or ('junction' in s) or (
            'transport' in s) or ('crossing' in s)
            or ('access' in s) or ('4wd_only' in s) or ('minor' in s) or ('park' in s) or ('barrier' in s) or (
                    'guidepost' in s) or ('restriction' in s)
            or ('entrance' in s) or ('foot=no' in s) or ('direction' in s) or ('stop' in s) or ('kerb' in s) or (
                    'asphalt' in s) or ('checkpoint' in s)
            or ('maxspeed' in s) or ('description' in s) or ('ref' in s) or ('oneway' in s) or ('note' in s) or (
                    'speed' in s) or ('flashing_light' in s)
            or ('surface' in s) or ('way' in s) or ('left' in s) or ('right' in s) or ('lanes' in s) or (
                    'milestone' in s) or ('sidewalk' in s)
            or ('hazard' in s) or ('way' in s) or ('left' in s) or ('right' in s) or ('lanes' in s) or (
                    'milestone' in s) or ('sidewalk' in s)):
        return 0  # 'traffic'
    elif ('restaurant' in s) or ('food' in s):
        return 1  # 'restaurant'
    elif ('university' in s) or ('school' in s):
        return 2  # 'education'
    elif (('power' in s) or ('bridge' in s) or ('building' in s) or ('pipeline' in s) or ('toilets' in s) or (
            'lit' in s) or ('fire_hydrant' in s)
          or ('height' in s) or ('amenity' in s) or ('shop' in s) or ('amenity' in s) or ('amenity' in s)):
        return 3  # 'facilities'
    elif (('ford' in s) or ('natural' in s) or ('forest' in s) or ('lake' in s) or ('wood' in s) or ('log' in s) or (
            'river' in s)
          or ('hill' in s) or ('valley' in s) or ('mountain' in s) or ('water' in s) or ('natural' in s) or (
                  'natural' in s) or ('natural' in s)):
        return 4  # 'nature'
    elif (('bus' in s) or ('bicycle' in s) or ('ferry_terminal' in s) or ('airport' in s) or ('aeroway' in s) or (
            'tourism' in s) or ('airport' in s)
          or ('airport' in s) or ('airport' in s) or ('airport' in s) or ('airport' in s) or ('airport' in s) or (
                  'airport' in s)):
        return 5  # 'transportation'
    elif (('leisure' in s) or ('fountain' in s) or ('square' in s) or ('club' in s) or ('bar' in s) or (
            'aerialway' in s) or ('aerialway' in s) or ('aerialway' in s)
          or ('aerialway' in s) or ('aerialway' in s) or ('aerialway' in s) or ('aerialway' in s) or (
                  'aerialway' in s)):
        return 6  # 'amusement'
    elif (('hamlet' in s) or ('village' in s) or ('town' in s) or ('city' in s) or ('populated place' in s) or (
            'neighbourhood' in s) or ('city' in s)):
        return 7  # 'populated place'
    elif 'ele' in s:
        return 8  # 'unknow_eles'
    elif (('man_made' in s) or ('historic' in s) or ('name' in s) or ('historic' in s) or ('curve_geometry' in s) or (
            'addr' in s) or ('layer' in s)
          or ('postal_code' in s) or ('level' in s) or ('curve_geometry' in s)):
        return 9  # 'man_made'
    elif (('source' in s) or ('created' in s) or ('attribution' in s) or ('odbl' in s) or ('tiger' in s) or (
            'website' in s) or ('wikidata' in s)):
        return 10  # 'attributor'
    elif (('tmp' in s) or ('fixme' in s)):
        return None
    else:
        return 11  # 'unknown'


data['tags'] = data['tags'].str.lower()
data['tags'] = data['tags'].apply(classify)
data = data.dropna()
data = data.drop('id', axis=1)
'''
data_ex=data.explode('def_tags')
tag_counts = data_ex['def_tags'].value_counts()

print(tag_counts.head(20))
'''
# print(data)
data.to_csv('classed_poi.csv', index=False)
