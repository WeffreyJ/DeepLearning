import csv, os
class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path; self.fieldnames = fieldnames
        if not os.path.exists(self.path):
            with open(self.path,'w',newline='',encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
    def log(self, row):
        with open(self.path,'a',newline='',encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
