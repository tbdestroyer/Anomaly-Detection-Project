import psutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import time
import threading

class ResourceMonitor:
    def __init__(self):
        self.output_dir = 'outputs/monitoring'
        os.makedirs(self.output_dir, exist_ok=True)
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval=60):
        """Start monitoring system resources"""
        if self.monitoring:
            print("Monitoring is already running")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started resource monitoring with {interval}s interval")
        
    def stop_monitoring(self):
        """Stop monitoring system resources"""
        if not self.monitoring:
            print("Monitoring is not running")
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Stopped resource monitoring")
        
    def _monitor_resources(self, interval):
        """Monitor system resources at specified interval"""
        while self.monitoring:
            metrics = self._collect_metrics()
            self._save_metrics(metrics)
            self._check_resource_alerts(metrics)
            time.sleep(interval)
            
    def _collect_metrics(self):
        """Collect system resource metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }
        
    def _save_metrics(self, metrics):
        """Save metrics to CSV"""
        df = pd.DataFrame([metrics])
        if os.path.exists(f'{self.output_dir}/resource_metrics.csv'):
            df.to_csv(f'{self.output_dir}/resource_metrics.csv', mode='a', header=False, index=False)
        else:
            df.to_csv(f'{self.output_dir}/resource_metrics.csv', index=False)
            
    def _check_resource_alerts(self, metrics):
        """Check for resource usage alerts"""
        alerts = []
        
        # CPU alert
        if metrics['cpu_usage'] > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'resource': 'CPU',
                'usage': metrics['cpu_usage'],
                'threshold': 90
            })
            
        # Memory alert
        if metrics['memory_usage'] > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'resource': 'Memory',
                'usage': metrics['memory_usage'],
                'threshold': 90
            })
            
        # Disk alert
        if metrics['disk_usage'] > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'resource': 'Disk',
                'usage': metrics['disk_usage'],
                'threshold': 90
            })
            
        if alerts:
            # Save alerts
            with open(f'{self.output_dir}/resource_alerts.json', 'w') as f:
                json.dump(alerts, f)
                
            # Print alerts
            for alert in alerts:
                print(f"ALERT: {alert['resource']} usage at {alert['usage']}% (threshold: {alert['threshold']}%)")
                
    def visualize_resource_usage(self, time_range='24h'):
        """Create visualizations for resource usage"""
        try:
            df = pd.read_csv(f'{self.output_dir}/resource_metrics.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time range
            if time_range == '1h':
                df = df[df['timestamp'] > datetime.now() - pd.Timedelta(hours=1)]
            elif time_range == '6h':
                df = df[df['timestamp'] > datetime.now() - pd.Timedelta(hours=6)]
            elif time_range == '24h':
                df = df[df['timestamp'] > datetime.now() - pd.Timedelta(hours=24)]
                
            if len(df) == 0:
                print("No data available for the selected time range")
                return
                
            # Create resource usage plot
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1)
            plt.plot(df['timestamp'], df['cpu_usage'])
            plt.title('CPU Usage')
            plt.xlabel('Time')
            plt.ylabel('Usage (%)')
            plt.axhline(y=90, color='r', linestyle='--')
            
            plt.subplot(3, 1, 2)
            plt.plot(df['timestamp'], df['memory_usage'])
            plt.title('Memory Usage')
            plt.xlabel('Time')
            plt.ylabel('Usage (%)')
            plt.axhline(y=90, color='r', linestyle='--')
            
            plt.subplot(3, 1, 3)
            plt.plot(df['timestamp'], df['disk_usage'])
            plt.title('Disk Usage')
            plt.xlabel('Time')
            plt.ylabel('Usage (%)')
            plt.axhline(y=90, color='r', linestyle='--')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/resource_usage.png')
            plt.close()
            
        except FileNotFoundError:
            print("No resource metrics found") 