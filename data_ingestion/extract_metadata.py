import psycopg2
from utils.config import PG_CONFIG

def store_metadata(application_number, project_name, project_type, location, status, materials, date):
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    query = """
    INSERT INTO planning_metadata (application_number, project_name, project_type, location, decision_status, materials, date_of_decision)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (application_number) DO NOTHING;
    """

    cur.execute(query, (application_number, project_name, project_type, location, status, materials, date))
    conn.commit()
    cur.close()
    conn.close()
