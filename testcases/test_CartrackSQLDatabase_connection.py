
import os
from ..src.Database.CartrackSQLDatabase import CartrackSQLDatabase

database =[
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "dbreporting.cartrack.co.za",
        'port' : 5432,
        'database_name': "cartrack"
    },
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "fleetreporting.cartrack.co.za",
        'port' : 5432,
        'database_name': "ct_fleet"
    },
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "new_dev.cartrack.co.za",
        'port' : 5432,
        'database_name': "CARTRACK_UAT"
    },
]



if __name__ == "__main__":
    custom_db = CartrackSQLDatabase.from_uri_config(database[2])
    