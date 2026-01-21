from config.security_config import MIN_CONFIDENCE_SCORE, FACE_MATCH_THRESHOLD

class ConfidenceEngine:
    """Centralized decision logic for authentication."""
    
    def calculate_login_score(self, distances, liveness_results, challenge_passed):
        """Aggregates all signals into a single confidence score (0-1)."""
        if not distances:
            return 0.0, "No face match"
        
        # 1. Matching Score
        # distance is 0 for perfect match, ~0.6 for threshold
        # We invert it: 1.0 for perfect match, 0 for threshold
        best_distance = min(distances)
        match_score = max(0, 1 - (best_distance / (FACE_MATCH_THRESHOLD * 1.5)))
        
        # 2. Liveness Score
        liveness_score = liveness_results.get("liveness_score", 0.0)
        
        # 3. Overall Aggregation
        # Challenge pass is critical - if failed, total score drops significantly
        final_score = (match_score * 0.5) + (liveness_score * 0.3)
        if challenge_passed:
            final_score += 0.2
        else:
            final_score *= 0.1 # Severe penalty for failing challenge
            
        decision = final_score >= MIN_CONFIDENCE_SCORE
        
        reason = "Success" if decision else "Insufficient confidence"
        if not challenge_passed:
            reason = "Liveness challenge failed"
        elif best_distance > FACE_MATCH_THRESHOLD:
            reason = "Face match failed"
            
        return final_score, reason

    def is_access_granted(self, score):
        return score >= MIN_CONFIDENCE_SCORE
