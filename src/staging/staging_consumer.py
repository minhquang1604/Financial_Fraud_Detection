import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
STAGING_TOPIC = "transaction_events"
STAGING_OUTPUT_DIR = "data/staging"


class StagingConsumer:
    def __init__(
        self,
        bootstrap_servers: str = BOOTSTRAP_SERVERS,
        topic: str = STAGING_TOPIC,
        output_dir: str = STAGING_OUTPUT_DIR,
        batch_size: int = 1000,
        group_id: str = "staging-consumer"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.group_id = group_id
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.staging_data = []
        self.offset_file = os.path.join(output_dir, "last_offset.json")
        
        self.last_offset = self._load_last_offset()
    
    def _load_last_offset(self) -> int:
        if os.path.exists(self.offset_file):
            with open(self.offset_file, 'r') as f:
                data = json.load(f)
                return data.get("offset", -1)
        return -1
    
    def _save_last_offset(self, offset: int):
        with open(self.offset_file, 'w') as f:
            json.dump({"offset": offset, "timestamp": datetime.now().isoformat()}, f)
    
    def create_consumer(self) -> KafkaConsumer:
        return KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            group_id=self.group_id,
            client_id='staging-consumer'
        )
    
    def process_message(self, message) -> Optional[Dict[str, Any]]:
        try:
            data = message.value
            
            data["_kafka_metadata"] = {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
                "timestamp": message.timestamp,
                "received_at": datetime.now().isoformat()
            }
            
            return data
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def save_batch(self, data: List[Dict[str, Any]]):
        if not data:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.output_dir, f"staging_batch_{timestamp}.parquet")
        
        df = pd.DataFrame(data)
        df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(data)} records to {filepath}")
        
        if data:
            last_record = data[-1]
            last_offset = last_record.get("_kafka_metadata", {}).get("offset", 0)
            self._save_last_offset(last_offset)
    
    def run(self):
        logger.info(f"Starting staging consumer for topic '{self.topic}'...")
        
        consumer = self.create_consumer()
        
        if self.last_offset >= 0:
            logger.info(f"Seeking to offset {self.last_offset + 1}")
            consumer.seek(offset=self.last_offset + 1)
        
        try:
            for message in consumer:
                processed = self.process_message(message)
                
                if processed:
                    self.staging_data.append(processed)
                    
                    if len(self.staging_data) >= self.batch_size:
                        self.save_batch(self.staging_data)
                        self.staging_data = []
                        
                        consumer.commit()
                        
                        logger.info(f"Batch saved, committed offset")
                
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        finally:
            if self.staging_data:
                self.save_batch(self.staging_data)
            
            consumer.close()
            logger.info("Consumer closed")

    def get_staging_files(self) -> List[str]:
        if not os.path.exists(self.output_dir):
            return []
        
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.parquet')]
        return sorted([os.path.join(self.output_dir, f) for f in files])


def run_staging_consumer():
    consumer = StagingConsumer()
    consumer.run()


if __name__ == "__main__":
    run_staging_consumer()