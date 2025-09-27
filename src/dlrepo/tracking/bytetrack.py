import numpy as np
def iou(a, b):
    xx1 = np.maximum(a[0], b[0]); yy1 = np.maximum(a[1], b[1])
    xx2 = np.minimum(a[2], b[2]); yy2 = np.minimum(a[3], b[3])
    w = np.maximum(0., xx2-xx1); h = np.maximum(0., yy2-yy1)
    inter = w*h; ra = (a[2]-a[0])*(a[3]-a[1]); rb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ra + rb - inter + 1e-9)
class ByteTrackLite:
    def __init__(self, conf_th=0.3, match_th=0.7, max_age=30):
        self.conf_th, self.match_th, self.max_age = conf_th, match_th, max_age
        self.next_id = 1; self.tracks = {}  # id -> dict(box, cls, score, age)
    def update(self, detections, t):
        dets = [d for d in (detections or []) if d[4] >= self.conf_th]
        for tr in self.tracks.values(): tr["age"] += 1
        used=set()
        for tid, tr in list(self.tracks.items()):
            best=-1; besti=-1
            for i,d in enumerate(dets):
                if i in used: continue
                ov = iou(tr["box"], d[:4])
                if ov>best: best, besti = ov, i
            if best>=self.match_th:
                d = dets[besti]; used.add(besti)
                tr["box"], tr["score"], tr["cls"], tr["age"] = d[:4], d[4], d[5], 0
        for i,d in enumerate(dets):
            if i in used: continue
            self.tracks[self.next_id] = {"box": d[:4], "score": d[4], "cls": d[5], "age": 0}
            self.next_id += 1
        self.tracks = {k:v for k,v in self.tracks.items() if v["age"] <= self.max_age}
        out=[]
        for tid,tr in self.tracks.items():
            x1,y1,x2,y2 = tr["box"]
            out.append([tid, x1,y1,x2,y2, tr["score"], tr["cls"]])
        return out
