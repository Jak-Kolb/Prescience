
class UniqueTrackCounter:
    def __init__(self, min_hits: int = 5):
        self.min_hits = min_hits
        self.total = 0
        self._seen_frames: dict[int, int] = {} # track_id -> number of hits
        self._counted: set[int] = set() # track_ids that have already been counted
        
    def update(self, track_id: int) -> int:
        """Update the counter with teh active track IDs 
        for THIS frame
        
        returns number of new countes added to this frame (0,1,2,...)"""
        
        new_count = 0
        
        for tid in track_id:
            self._seen_frames[tid] = self._seen_frames.get(tid, 0) + 1
            
            if self._seen_frames[tid] >= self.min_hits and tid not in self._counted:
                self.total += 1
                new_count += 1
                self._counted.add(tid)
                print(f"Counted track ID {tid}. Total count: {self.total}")
                
        return new_count