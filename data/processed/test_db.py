import psycopg2

conn = psycopg2.connect(
    dbname="inflation_ai",
    user="postgres",
    password="golaghat@123",
    host="localhost",
    port="5432"
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM inflation_data;")

print(cur.fetchone())

cur.close()
conn.close()