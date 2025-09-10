def generate_crowd_alert(results, count_threshold=50, avg_threshold=30):
    """
    Generate a unified alert based on the analysis results.
    Returns a warning message if crowd levels are high.
    """
    # Calculate risk score
    risk_score = 0
    max_count = results['max_count']
    avg_count = results['average_count']
    high_density_ratio = results['density_levels'].count('HIGH') / len(results['density_levels']) if results['density_levels'] else 0
    
    # Assess maximum crowd count
    if max_count >= count_threshold or high_density_ratio > 0.5:
        risk_score += 2
    elif max_count >= 30 or high_density_ratio > 0.3:
        risk_score += 1
        
    # Assess average crowd count
    if avg_count >= avg_threshold:
        risk_score += 2
    elif avg_count >= 20:
        risk_score += 1
    
    # Generate alert message based on risk score
    if risk_score >= 4:
        return "🚨 **HIGH RISK: Critical crowd activity detected.**\n\n📹 **Immediate Action Required:** Please review the full footage for potential emergencies or unsafe conditions."
    elif risk_score >= 2:
        return "⚠️ **MODERATE RISK: Elevated crowd levels observed.**\n\n📹 **Recommendation:** Review specific intervals for safety assessment."
    else:
        return "✅ **LOW RISK: Crowd activity appears normal.** No immediate action required."