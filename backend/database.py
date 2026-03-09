from pymongo import MongoClient as mc
from datetime import datetime as dt
import configure as cfg

class Database:
    def __init__ (self):
        try:
            self.client= mc(cfg.MONGO_URI)
            self.db = self.client[cfg.DB_NAME]
            self.logs_collection = self.db[cfg.COLLECTION_LOGS]
            self.known_faces_collection = self.db[cfg.COLLECTION_KNOWN_FACES]
            print ("Connected to MongoDB successfully")
        except Exception as e:
         print ("Error connecting to MongoDB:", e)

    def log_unknown_detecttion(self. image_path , timestamp=None):
        try :
           if timestamp is None:
              timestamp = dt.now()
              log_entry ={
               "type": "unknown_face",
               "timestamp": timestamp,
               "date": timestamp.strftime("%Y-%m-%d"),
               "time": timestamp.strftime("%H:%M:%S"),
            }
           result = self.logs_collection.insert_one(log_entry)
           return str(result.inserted_id)
        except Exception as e:
            print ("Error logging unknown detection:", e)
            return None
    def log_knwon_detection(self, name , image_path,timestamp=None):
       try:
          if timestamp is None:
             timestamp = dt.now()
             log_entry ={
                "type": "known_face",
                "name": name,
                "timestamp": timestamp,
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
             }
             result = self.logs_collection.insert_one(log_entry)
             return str(result.inserted_id)
       except Exception as e:
          print ("Error logging known detection:", e)
          return None
    def get_unknown_count_today(self):
        try:
            today = dt.now().strftime("%Y-%m-%d")
            count = self.logs_collection.count_documents({
               "type": "unknown_face",
               "date" : today
            })
            return count
        except Exception as e:
            print ("Error getting unknown count for today:", e)
            return 0
    def get_unknown_count_total(self):
        try:
            count = self.logs_collection.count_documents({
               "type":"unknown_face"
            })
            return count
        except Exception as e :
            print ("Error getting total unknown count:", e)
            return 0
    def get_recent_detections(self, limit=10):
        try:
            detections = self.logs_collections.find().sort("timestamp", -1).limit(limit)
            return list(detections)
        except Exception as e:  
            print ("Error getting recent detections:", e)
            return []
    def register_known_face(self,name,face_encoding,image_path):
        try:
            face_data = {
                "name": name,
                "face_encoding": face_encoding.tolist(),
                "image_path": image_path,
                "registered_at": dt.now()
            }
            result = self.known_faces_collection.insert_one(face_data)
            return str(result.inserted_id)
        except Exception as e:
            print ("Error registering known face:", e)
            return None
    def get_all_known_faces(self):
        try:
            faces = self.known_faces_collection.find()
            return list(faces)
        except Exception as e:
            print ("Error getting all known faces:", e)
            return []
    def delete_known_face(self,name):
        try:
            result= self.known_faces_collection.delete_one({
                "name": name
            })
            return result.deleted_count>0
        except Exception as e:  
            print ("Error deleting known face:", e)
            return False
        
db= Database()


            

