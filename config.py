

class Config():
    path = "D:\\ABH Internship\\dataset"
    sql_table_name = "business"
    db_name = "dataset"
    db_user = "postgres"
    db_password = "kkk123"


    standardized_dict_yelp = {"restaurants": "restaurant", "beauty & spas": "beauty_salon",
                              "nail salons": "beauty_salon", "shopping": "store", "health & medical": "health",
                              "home services": "moving_company, painter, electrician, storage, plumber, home_goods_store",
                              "local services": "moving_company, painter, electrician, storage, plumber, car_repair, car_wash",
                              "automotive": "car_repair, car_wash, gas_station, car_rental", "doctors": "doctor",
                              "active life": "gym, spa, amusement_park, park, active_life",
                              "professional services": "plumber, electrician, accounting, lawyer",
                              "real estate": "lodging", "nightlife": "night_club, bar",
                              "home & garden": "home_goods_store, furniture_store", "bars": "bar", "hair salons": "hair_care",
                              "fashion": "clothing_store", "auto repair": "car_repair", "fast food": "food",
                              "mexican": "food", "sandwiches": "food", "american (traditional)": "food",
                              "fitness & instruction": "gym", "dentists": "dentist", "coffee & tea": "cafe",
                              "hotels & travel": "lodging, travel_agency",
                              "arts & entertainment": "museum, night_club, amusement_park, aquarium, park",
                              "financial services": "bank, accounting", "pizza": "food", "hair removal": "beauty_salon",
                              "american (new)": "food", "education": "school", "pets": "pet_store", "skin care": "beauty_salon",
                              "general dentistry": "dentist", "burgers": "food", "breakfast & brunch": "food",
                              "apartments": "lodging", "cosmetic dentists": "dentist", "pet services": "veterinary_care",
                              "italian": "food", "grocery": "home_goods_store", "waxing": "beauty_salon",
                              "womens clothing": "clothing_store", "hair stylists": "hair_care", "massage": "beauty_salon",
                              "gyms": "gym", "cafes": "cafe", "gas stations": "gas_station", "furniture stores": "furniture_store",
                              "banks & credit unions": "bank", "post offices": "post_office",
                              "shopping centers": "shopping_mall", "jewelry" :"jewelry_store"}

    standardized_dict_osm = {"swimming_pool": "spa", "fast_food": "food", "convenience": "convenience_store",
                             "supermarket": "store, supermarket", "hotel": "lodging", "motel": "lodging",
                             "clothes": "clothing_store", "beauty_shop": "beauty_salon", "pub": "pub, bar",
                             "hairdresser": "hair_care", "doctors": "doctor, hospital", "sports_centre": "active_life",
                             "furniture_shop": "furniture_store", "veterinary": "veterinary_care",
                             "beverages": "bar, cafe", "guesthouse": "lodging", "mall": "shopping_mall",
                             "nightclub": "night_club", "college": "school", "shoe_shop": "store, clothing_store",
                             "theme_park":"amusement_park", "jeweller": "jewelry_store", "hostel": "lodging"}