import sqlite3
conn = sqlite3.connect("ROI.db",check_same_thread=False)

c = conn.cursor()
class UserClass:
    def __init__(self):
        pass
    def create_table(self):
        with conn:
            c.execute('''CREATE TABLE roi_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            lane_number INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
        
    def insert_data(self,video_name,lane_number,x,y,width,height):
        with conn:
            c.execute("INSERT INTO roi_data (video_name,lane_number,x,y,width,height) VALUES (?,?,?,?,?,?)",(video_name,lane_number,x,y,width,height))        

# c.execute(f"SELECT * from roi_data")
# result = c.fetchall()
# print(result)