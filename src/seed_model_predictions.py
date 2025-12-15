"""
Seed model_predictions with realistic physics-based predictions
"""
from db_connect import fetch_dataframe, execute_query, bulk_insert
from synthetic_predictor import RealisticPredictor
from config_file import Config
from datetime import datetime, timedelta
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_sensor_reading(base_health: float = 0.5):
    """
    Generate realistic sensor readings
    base_health: 0=critical, 0.5=medium, 1=healthy
    """
    # Healthy ranges
    healthy_ranges = {
        'air_temperature': (20, 30),
        'process_temperature': (30, 42),
        'rotational_speed': (1400, 1600),
        'torque': (35, 50),
        'tool_wear': (20, 150)
    }
    
    # Critical ranges
    critical_ranges = {
        'air_temperature': (35, 42),
        'process_temperature': (48, 56),
        'rotational_speed': (1850, 2000),
        'torque': (70, 85),
        'tool_wear': (220, 260)
    }
    
    features = {}
    
    for key in healthy_ranges.keys():
        healthy_min, healthy_max = healthy_ranges[key]
        critical_min, critical_max = critical_ranges[key]
        
        # Interpolate between healthy and critical based on base_health
        range_min = healthy_min + (critical_min - healthy_min) * (1 - base_health)
        range_max = healthy_max + (critical_max - healthy_max) * (1 - base_health)
        
        features[key] = random.uniform(range_min, range_max)
    
    return features


def main():
    """Generate realistic predictions for all machines"""
    
    # Clear existing predictions
    try:
        existing = execute_query("SELECT COUNT(*) FROM model_predictions", fetch=True)[0][0]
        if existing > 0:
            logger.info(f"Found {existing} existing predictions")
            response = input("Clear and regenerate? (yes/no): ").strip().lower()
            if response != 'yes':
                logger.info("Seeding cancelled")
                return
            execute_query("DELETE FROM model_predictions")
            logger.info("✓ Cleared old predictions")
    except Exception as e:
        logger.error(f"Could not query model_predictions: {e}")
        return

    # Initialize predictor
    predictor = RealisticPredictor()
    
    # Load machines
    try:
        machines = fetch_dataframe("SELECT machine_id FROM machines ORDER BY machine_id")
    except Exception as e:
        logger.error(f"Could not load machines: {e}")
        return
    
    if len(machines) == 0:
        logger.warning("No machines found – nothing to predict")
        return

    rows = []
    
    logger.info(f"\n📊 Generating predictions for {len(machines)} machines...")
    
    for machine_idx, r in enumerate(machines.itertuples(index=False), 1):
        machine_id = int(r.machine_id)
        
        # Assign health profile to each machine
        # 60% healthy, 25% medium, 15% critical
        rand = random.random()
        if rand < 0.60:
            base_health = random.uniform(0.7, 1.0)  # Healthy
            health_label = "Healthy"
        elif rand < 0.85:
            base_health = random.uniform(0.4, 0.7)  # Medium risk
            health_label = "Medium Risk"
        else:
            base_health = random.uniform(0.0, 0.4)  # Critical
            health_label = "Critical"
        
        logger.info(f"\nMachine {machine_id} ({health_label}):")
        
        # Generate 50 historical predictions showing degradation
        for days_ago in range(50, 0, -1):
            timestamp = datetime.now() - timedelta(days=days_ago)
            
            # Simulate gradual degradation over time
            time_factor = (50 - days_ago) / 50.0  # 0 to 1 over 50 days
            current_health = base_health * (1 - time_factor * 0.3)  # Degrade by up to 30%
            
            # Generate sensor readings
            features = generate_realistic_sensor_reading(current_health)
            
            # Get prediction
            try:
                result = predictor.predict_all(features)
                
                failure_prob = result['failure_prediction']['failure_probability']
                predicted_rul = result['rul_prediction']['predicted_rul_days']
                anomaly_score = result['anomaly_detection']['anomaly_score']
                
                rows.append((
                     int(machine_id),
                     timestamp,
                     float(failure_prob),
                     float(predicted_rul),
                     float(anomaly_score),
                     'physics_based_v1.0'
                ))
                
                # Log latest prediction
                if days_ago == 1:
                    logger.info(f"  Latest: Failure={failure_prob:.2%}, RUL={predicted_rul:.1f}d")
                
            except Exception as e:
                logger.error(f"  Prediction failed for machine {machine_id}: {e}")
                continue

    if not rows:
        logger.info("No predictions generated")
        return

    # Insert into DB
    try:
        logger.info(f"\n💾 Inserting {len(rows)} predictions...")
        
        bulk_insert(
            'model_predictions',
            ['machine_id', 'prediction_timestamp', 'failure_probability', 
             'predicted_rul', 'anomaly_score', 'model_version'],
            rows,
            page_size=500
        )
        
        logger.info(f"✅ Inserted {len(rows)} predictions successfully!")
        
        # Show statistics
        stats_query = """
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT machine_id) as machines,
            ROUND(AVG(failure_probability)::numeric, 4) as avg_failure,
            ROUND(MIN(failure_probability)::numeric, 4) as min_failure,
            ROUND(MAX(failure_probability)::numeric, 4) as max_failure,
            ROUND(AVG(predicted_rul)::numeric, 2) as avg_rul,
            ROUND(MIN(predicted_rul)::numeric, 2) as min_rul,
            ROUND(MAX(predicted_rul)::numeric, 2) as max_rul
        FROM model_predictions
        """
        stats = fetch_dataframe(stats_query)
        
        logger.info("\n📊 Statistics:")
        s = stats.iloc[0]
        logger.info(f"  Total predictions: {s['total']}")
        logger.info(f"  Machines: {s['machines']}")
        logger.info(f"  Failure probability: {float(s['min_failure']):.2%} to {float(s['max_failure']):.2%}")
        logger.info(f"  Average failure: {float(s['avg_failure']):.2%}")
        logger.info(f"  RUL: {float(s['min_rul']):.1f} to {float(s['max_rul']):.1f} days")
        logger.info(f"  Average RUL: {float(s['avg_rul']):.1f} days")
        
        # Check distribution
        dist_query = """
        SELECT 
            CASE 
                WHEN failure_probability < 0.3 THEN 'Low Risk'
                WHEN failure_probability < 0.6 THEN 'Medium Risk'
                ELSE 'High Risk'
            END as risk_category,
            COUNT(*) as count,
            ROUND(COUNT(*)::numeric / (SELECT COUNT(*) FROM model_predictions) * 100, 1) as percentage
        FROM model_predictions
        WHERE prediction_timestamp = (SELECT MAX(prediction_timestamp) FROM model_predictions)
        GROUP BY risk_category
        ORDER BY risk_category
        """
        dist = fetch_dataframe(dist_query)
        
        logger.info("\n📈 Latest Predictions Distribution:")
        for _, row in dist.iterrows():
            logger.info(f"  {row['risk_category']}: {row['count']} machines ({row['percentage']}%)")
        
        # Check for 99% problem
        high_query = """
        SELECT COUNT(*) as cnt 
        FROM model_predictions 
        WHERE failure_probability > 0.99
        """
        high_df = fetch_dataframe(high_query)
        high_count = high_df.iloc[0]['cnt']
        
        if high_count > 0:
            logger.warning(f"\n⚠️  Found {high_count} predictions with >99% failure")
        else:
            logger.info("\n✅ No unrealistic 99% predictions found!")
        
    except Exception as e:
        logger.error(f"❌ Failed to insert predictions: {e}")


if __name__ == '__main__':
    main()