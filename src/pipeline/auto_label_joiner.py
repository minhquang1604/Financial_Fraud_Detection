import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from label_joiner import LabelJoiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STAGING_DIR = os.path.join(PROJECT_ROOT, "data", "staging")
PROCESS_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LABELED_DIR = os.path.join(PROJECT_ROOT, "data", "labeled")


class StagingFileHandler(FileSystemEventHandler):
    def __init__(self, joiner: LabelJoiner, batch_size: int = 1000):
        self.joiner = joiner
        self.batch_size = batch_size
        self.last_processed = {}
        self.processing = False
        self.cooldown_seconds = 5
        
    def _is_new_file(self, filepath: str) -> bool:
        if not filepath.endswith('.parquet'):
            return False
        if 'staging_batch' not in filepath:
            return False
        if filepath in self.last_processed:
            last_time = self.last_processed[filepath]
            if (datetime.now() - last_time).total_seconds() < self.cooldown_seconds:
                return False
        return True
    
    def _process_batch(self):
        if self.processing:
            return
        self.processing = True
        try:
            result = self.joiner.process_batch()
            logger.info(f"Auto label join result: {result}")
        except Exception as e:
            logger.error(f"Error in auto label join: {e}")
        finally:
            self.processing = False
    
    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        filepath = event.src_path
        if self._is_new_file(filepath):
            logger.info(f"New staging file detected: {filepath}")
            self.last_processed[filepath] = datetime.now()
            time.sleep(1)
            self._process_batch()
    
    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        filepath = event.src_path
        if self._is_new_file(filepath):
            logger.info(f"Staging file modified: {filepath}")
            self.last_processed[filepath] = datetime.now()
            self._process_batch()


class AutoLabelJoiner:
    def __init__(
        self,
        staging_dir: str = STAGING_DIR,
        watch: bool = True,
        poll_interval: int = 60
    ):
        self.staging_dir = staging_dir
        self.watch_mode = watch
        self.poll_interval = poll_interval
        self.joiner = LabelJoiner()
        self.running = False
        
    def start_file_watcher(self):
        if not os.path.exists(self.staging_dir):
            os.makedirs(self.staging_dir, exist_ok=True)
            logger.info(f"Created staging directory: {self.staging_dir}")
        
        event_handler = StagingFileHandler(self.joiner)
        observer = Observer()
        observer.schedule(event_handler, self.staging_dir, recursive=False)
        observer.start()
        logger.info(f"Started file watcher on {self.staging_dir}")
        
        return observer
    
    def start_polling(self):
        logger.info(f"Started polling mode (interval: {self.poll_interval}s)")
        while self.running:
            try:
                result = self.joiner.process_batch()
                if result.get('success'):
                    logger.info(f"Polling join result: {result.get('records', 0)} records")
            except Exception as e:
                logger.error(f"Error in polling: {e}")
            time.sleep(self.poll_interval)
    
    def run(self, watch: Optional[bool] = None):
        mode = watch if watch is not None else self.watch_mode
        logger.info("=" * 60)
        logger.info("AUTO LABEL JOINER")
        logger.info(f"Mode: {'File Watcher' if mode else 'Polling'}")
        logger.info("=" * 60)
        
        self.running = True
        
        if mode:
            observer = self.start_file_watcher()
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.running = False
                observer.stop()
            observer.join()
        else:
            self.start_polling()
        
        logger.info("Auto label joiner stopped")


def run_auto_joiner(watch: bool = True, poll_interval: int = 60):
    auto_joiner = AutoLabelJoiner(watch=watch, poll_interval=poll_interval)
    auto_joiner.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Label Joiner")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="watch",
        choices=["watch", "poll"],
        help="Mode: watch (file system events) or poll (interval)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Poll interval in seconds (default: 60)"
    )
    args = parser.parse_args()
    
    run_auto_joiner(watch=(args.mode == "watch"), poll_interval=args.interval)