#!/usr/bin/env python3
"""
Real-time Network Traffic Feature Extraction and Anomaly Detection
Captures network packets, extracts CIC-IDS style features, and passes through VAE
"""

import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP
from threading import Thread, Lock
from catboost import CatBoost
import warnings
warnings.filterwarnings('ignore')

# Top features to extract
TOP_FEATURES = [
    'Fwd Pkts/s', 'Init Bwd Win Byts', 'Flow Pkts/s', 'Fwd Seg Size Min', 
    'Init Fwd Win Byts', 'Flow IAT Std', 'Pkt Len Max', 'ACK Flag Cnt', 
    'Fwd Header Len', 'Fwd Pkt Len Std', 'Bwd Pkts/s', 'Flow Byts/s', 
    'Fwd Pkt Len Max', 'Bwd Header Len', 'Fwd IAT Tot', 'Bwd Pkt Len Max', 
    'Fwd Pkt Len Mean', 'URG Flag Cnt', 'Fwd IAT Std', 'Pkt Len Std', 
    'Flow IAT Min', 'Flow IAT Mean', 'Down/Up Ratio'
]

# VAE Model Definition (must match trained model architecture)
# Input: 10 time steps × 23 features = 230 input dimensions
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

# Window size for VAE input (10 time steps)
WINDOW_SIZE = 10


class FlowTracker:
    """Tracks network flows and computes features"""
    
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'fwd_packets': [],
            'bwd_packets': [],
            'fwd_timestamps': [],
            'bwd_timestamps': [],
            'fwd_header_lens': [],
            'bwd_header_lens': [],
            'init_fwd_win': None,
            'init_bwd_win': None,
            'ack_count': 0,
            'urg_count': 0,
            'start_time': None,
        })
        self.lock = Lock()
    
    def get_flow_key(self, pkt):
        """Generate flow key from packet"""
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            src_port = pkt.sport if TCP in pkt or UDP in pkt else 0
            dst_port = pkt.dport if TCP in pkt or UDP in pkt else 0
            # Ensure consistent flow direction
            if (src_ip, src_port) < (dst_ip, dst_port):
                return (src_ip, src_port, dst_ip, dst_port, proto)
            return (dst_ip, dst_port, src_ip, src_port, proto)
        return None
    
    def is_forward(self, pkt, flow_key):
        """Determine if packet is forward direction"""
        if IP in pkt:
            src_ip = pkt[IP].src
            src_port = pkt.sport if TCP in pkt or UDP in pkt else 0
            return (src_ip, src_port) == (flow_key[0], flow_key[1])
        return True
    
    def process_packet(self, pkt):
        """Process a captured packet"""
        flow_key = self.get_flow_key(pkt)
        if flow_key is None:
            return
        
        with self.lock:
            flow = self.flows[flow_key]
            timestamp = time.time()
            
            if flow['start_time'] is None:
                flow['start_time'] = timestamp
            
            pkt_len = len(pkt)
            is_fwd = self.is_forward(pkt, flow_key)
            
            if TCP in pkt:
                header_len = pkt[TCP].dataofs * 4
                flags = pkt[TCP].flags
                
                if 'A' in str(flags):
                    flow['ack_count'] += 1
                if 'U' in str(flags):
                    flow['urg_count'] += 1
                
                win_size = pkt[TCP].window
                if is_fwd and flow['init_fwd_win'] is None:
                    flow['init_fwd_win'] = win_size
                elif not is_fwd and flow['init_bwd_win'] is None:
                    flow['init_bwd_win'] = win_size
            else:
                header_len = 20  # Default IP header
            
            if is_fwd:
                flow['fwd_packets'].append(pkt_len)
                flow['fwd_timestamps'].append(timestamp)
                flow['fwd_header_lens'].append(header_len)
            else:
                flow['bwd_packets'].append(pkt_len)
                flow['bwd_timestamps'].append(timestamp)
                flow['bwd_header_lens'].append(header_len)
    
    def compute_features(self):
        """Compute features from current flows"""
        with self.lock:
            if not self.flows:
                return None
            
            all_features = []
            
            for flow_key, flow in self.flows.items():
                features = {}
                
                fwd_pkts = flow['fwd_packets']
                bwd_pkts = flow['bwd_packets']
                fwd_ts = flow['fwd_timestamps']
                bwd_ts = flow['bwd_timestamps']
                
                total_fwd = len(fwd_pkts)
                total_bwd = len(bwd_pkts)
                total_pkts = total_fwd + total_bwd
                
                if total_pkts == 0:
                    continue
                
                duration = max(
                    max(fwd_ts) if fwd_ts else 0,
                    max(bwd_ts) if bwd_ts else 0
                ) - flow['start_time']
                duration = max(duration, 0.001)  # Avoid division by zero
                
                # Packet rates
                features['Fwd Pkts/s'] = total_fwd / duration
                features['Bwd Pkts/s'] = total_bwd / duration
                features['Flow Pkts/s'] = total_pkts / duration
                
                # Bytes per second
                total_bytes = sum(fwd_pkts) + sum(bwd_pkts)
                features['Flow Byts/s'] = total_bytes / duration
                
                # Window sizes
                features['Init Fwd Win Byts'] = flow['init_fwd_win'] if flow['init_fwd_win'] else 0
                features['Init Bwd Win Byts'] = flow['init_bwd_win'] if flow['init_bwd_win'] else 0
                
                # Forward packet stats
                features['Fwd Pkt Len Max'] = max(fwd_pkts) if fwd_pkts else 0
                features['Fwd Pkt Len Mean'] = np.mean(fwd_pkts) if fwd_pkts else 0
                features['Fwd Pkt Len Std'] = np.std(fwd_pkts) if len(fwd_pkts) > 1 else 0
                features['Fwd Seg Size Min'] = min(fwd_pkts) if fwd_pkts else 0
                
                # Backward packet stats
                features['Bwd Pkt Len Max'] = max(bwd_pkts) if bwd_pkts else 0
                
                # Overall packet stats
                all_pkt_lens = fwd_pkts + bwd_pkts
                features['Pkt Len Max'] = max(all_pkt_lens) if all_pkt_lens else 0
                features['Pkt Len Std'] = np.std(all_pkt_lens) if len(all_pkt_lens) > 1 else 0
                
                # Header lengths
                features['Fwd Header Len'] = sum(flow['fwd_header_lens'])
                features['Bwd Header Len'] = sum(flow['bwd_header_lens'])
                
                # IAT (Inter-Arrival Time) features
                all_ts = sorted(fwd_ts + bwd_ts)
                if len(all_ts) > 1:
                    iats = np.diff(all_ts)
                    features['Flow IAT Std'] = np.std(iats)
                    features['Flow IAT Min'] = np.min(iats)
                    features['Flow IAT Mean'] = np.mean(iats)
                else:
                    features['Flow IAT Std'] = 0
                    features['Flow IAT Min'] = 0
                    features['Flow IAT Mean'] = 0
                
                # Forward IAT
                if len(fwd_ts) > 1:
                    fwd_iats = np.diff(sorted(fwd_ts))
                    features['Fwd IAT Tot'] = np.sum(fwd_iats)
                    features['Fwd IAT Std'] = np.std(fwd_iats)
                else:
                    features['Fwd IAT Tot'] = 0
                    features['Fwd IAT Std'] = 0
                
                # Flags
                features['ACK Flag Cnt'] = flow['ack_count']
                features['URG Flag Cnt'] = flow['urg_count']
                
                # Down/Up Ratio
                features['Down/Up Ratio'] = total_bwd / total_fwd if total_fwd > 0 else 0
                
                all_features.append(features)
            
            # Clear flows after computing
            self.flows.clear()
            
            return all_features


class RealTimeDetector:
    """Real-time anomaly detection using VAE + CatBoost"""
    
    def __init__(self, model_path='models/vae_full_model .pth', 
                 column_mapping_path='models/column_min_max_mapping.csv',
                 catboost_path='models/catboost_model.json',
                 top_features_path='models/top_features.joblib'):
        self.flow_tracker = FlowTracker()
        self.capturing = False
        
        # Window buffer: stores last 10 scaled feature vectors (each 23 features)
        # Initialize with zeros - shape: (WINDOW_SIZE, num_features)
        self.window_buffer = np.zeros((WINDOW_SIZE, len(TOP_FEATURES)))
        
        # Last valid feature values for ffill (forward fill)
        self.last_valid_features = None
        
        # Load model and scaler if available
        self.model = None
        self.column_mapping = None
        self.catboost_model = None
        self.top_features = None
        self.load_model(model_path, column_mapping_path, catboost_path, top_features_path)
    
    def load_model(self, model_path, column_mapping_path, catboost_path, top_features_path):
        """Load VAE model, column mapping, CatBoost model, and top features"""
        # Load column min-max mapping for scaling
        try:
            self.column_mapping = pd.read_csv(column_mapping_path)
            self.column_mapping = self.column_mapping.set_index('column')
            print(f"✓ Loaded column mapping from {column_mapping_path}")
        except Exception as e:
            print(f"⚠ Could not load column mapping: {e}")
        
        # Load top features
        try:
            self.top_features = joblib.load(top_features_path)
            print(f"✓ Loaded top features from {top_features_path}")
        except Exception as e:
            print(f"⚠ Could not load top features: {e}")
        
        try:
            # VAE input: 10 time steps × 23 features = 230
            self.model = VAE(input_dim=WINDOW_SIZE * len(TOP_FEATURES), latent_dim=5)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print(f"✓ Loaded VAE model from {model_path}")
        except Exception as e:
            print(f"⚠ Could not load VAE model: {e}")
            print("  Running in feature extraction mode only")
        
        try:
            self.catboost_model = CatBoost()
            self.catboost_model.load_model(catboost_path, format="json")
            print(f"✓ Loaded CatBoost model from {catboost_path}")
        except Exception as e:
            print(f"⚠ Could not load CatBoost model: {e}")
    
    def scale_features(self, df):
        """Scale features using min-max mapping to range 0-50"""
        if self.column_mapping is None:
            return df.values
        
        scaled = df.copy()
        for col in df.columns:
            if col in self.column_mapping.index:
                min_val = self.column_mapping.loc[col, 'min']
                max_val = self.column_mapping.loc[col, 'max']
                if max_val - min_val != 0:
                    # Scale to 0-50 range
                    scaled[col] = ((df[col] - min_val) / (max_val - min_val)) * 50
                else:
                    scaled[col] = 0
            else:
                # If column not in mapping, just normalize to 0-50 range
                if df[col].max() - df[col].min() != 0:
                    scaled[col] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) * 50
                else:
                    scaled[col] = 0
        return scaled.values
    
    def packet_callback(self, pkt):
        """Callback for packet capture"""
        self.flow_tracker.process_packet(pkt)
    
    def start_capture(self, interface=None):
        """Start packet capture in background thread"""
        self.capturing = True
        
        def capture_thread():
            print(f"Starting packet capture on interface: {interface or 'default'}")
            try:
                sniff(
                    iface=interface,
                    prn=self.packet_callback,
                    store=False,
                    stop_filter=lambda x: not self.capturing
                )
            except PermissionError:
                print("⚠ Permission denied. Run with sudo for packet capture.")
                self.capturing = False
            except Exception as e:
                print(f"Capture error: {e}")
                self.capturing = False
        
        thread = Thread(target=capture_thread, daemon=True)
        thread.start()
        return thread
    
    def process_features(self, features_list):
        """Process extracted features through VAE with sliding window of 10 time steps"""
        if not features_list:
            return None
        
        # Create DataFrame from captured flows
        df = pd.DataFrame(features_list)
        
        # Ensure all required features exist
        for feat in TOP_FEATURES:
            if feat not in df.columns:
                df[feat] = 0
        
        # Select only top features in correct order
        df = df[TOP_FEATURES]
        
        # Forward fill NaN values using last valid values
        if self.last_valid_features is not None:
            # Create a series from last valid values for ffill
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='ffill')
                    # If still NaN (first values), use last valid
                    if df[col].isna().any():
                        df[col] = df[col].fillna(self.last_valid_features[col])
        
        # Any remaining NaN fill with 0 (only for very first capture)
        df = df.fillna(0)
        
        # Aggregate all flows into single feature vector (mean of all flows in this interval)
        current_features = df.mean().values  # Shape: (23,)
        
        # Update last valid features for next iteration
        self.last_valid_features = pd.Series(current_features, index=TOP_FEATURES)
        
        # Scale features using column mapping (0-50 range)
        current_scaled = self.scale_single_features(current_features)
        
        # Shift window buffer: move everything back by 1, drop oldest
        # [0,1,2,3,4,5,6,7,8,9] -> [1,2,3,4,5,6,7,8,9,new]
        self.window_buffer[:-1] = self.window_buffer[1:]
        self.window_buffer[-1] = current_scaled  # Add current at the end
        
        # Flatten window for VAE input: (10, 23) -> (230,)
        vae_input = self.window_buffer.flatten()
        
        # Pass through VAE if available
        if self.model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(vae_input).unsqueeze(0)  # Shape: (1, 230)
                recon, mu, logvar, z = self.model(x)
                
                # Compute reconstruction loss
                recon_loss = torch.mean((x - recon) ** 2, dim=1).numpy()[0]
                
                # Compute KL divergence
                kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).numpy()[0]
                
                # Get latent features
                latent = z.numpy()[0]  # Shape: (5,)
                
                # Prepare features for CatBoost: latent_0-4, recon_loss, kld_loss
                catboost_features = np.array([[
                    latent[0], latent[1], latent[2], latent[3], latent[4],
                    recon_loss,
                    kld
                ]])  # Shape: (1, 7)
                
                # Predict with CatBoost if available
                prediction_label = None
                if self.catboost_model is not None:
                    # Get class predictions (not raw scores)
                    prediction = self.catboost_model.predict(catboost_features, prediction_type='Class')
                    prediction_label = int(prediction.flatten()[0])
                
                return {
                    'num_flows': len(features_list),
                    'current_features': current_features,
                    'latent': latent,
                    'recon_loss': recon_loss,
                    'kld_loss': kld,
                    'prediction_label': prediction_label
                }
        
        return {'num_flows': len(features_list), 'current_features': current_features}
    
    def scale_single_features(self, features):
        """Scale a single feature vector using min-max mapping to range 0-50"""
        if self.column_mapping is None:
            return features
        
        scaled = np.zeros_like(features)
        for i, col in enumerate(TOP_FEATURES):
            if col in self.column_mapping.index:
                min_val = self.column_mapping.loc[col, 'min']
                max_val = self.column_mapping.loc[col, 'max']
                if max_val - min_val != 0:
                    scaled[i] = ((features[i] - min_val) / (max_val - min_val)) * 50
                else:
                    scaled[i] = 0
            else:
                scaled[i] = 0
        return scaled
        
        return {'features': latest}
    
    def run(self, interface=None, interval=3, duration=300):
        """Main loop - capture and process every interval seconds for specified duration"""
        print("=" * 60)
        print("Real-Time Network Traffic Detection (CatBoost)")
        print("=" * 60)
        print(f"Capture interval: {interval} seconds")
        print(f"Total duration: {duration} seconds ({duration//60} minutes)")
        print(f"Features tracked: {len(TOP_FEATURES)}")
        print(f"Window size: {WINDOW_SIZE} time steps")
        print("-" * 60)
        
        # Start capture
        capture_thread = self.start_capture(interface)
        
        # Statistics for final report
        total_records = 0
        total_flows = 0
        total_benign = 0
        total_attacks = 0
        attack_classes = defaultdict(int)
        start_time = time.time()
        
        try:
            record_num = 0
            while self.capturing and (time.time() - start_time) < duration:
                time.sleep(interval)
                
                # Extract features from captured flows
                features_list = self.flow_tracker.compute_features()
                
                if features_list:
                    result = self.process_features(features_list)
                    
                    if result:
                        record_num += 1
                        total_records += 1
                        num_flows = result.get('num_flows', len(features_list))
                        total_flows += num_flows
                        
                        elapsed = time.time() - start_time
                        remaining = duration - elapsed
                        
                        print(f"\n[Record {record_num}] {time.strftime('%H:%M:%S')} | Remaining: {remaining:.0f}s")
                        print(f"  Flows in interval: {num_flows}")
                        
                        # Show CatBoost prediction (single prediction per interval)
                        if result.get('prediction_label') is not None:
                            label = result['prediction_label']
                            
                            if label == 0:
                                total_benign += 1
                                print(f"  ✓ BENIGN traffic")
                            else:
                                total_attacks += 1
                                attack_classes[label] += 1
                                print(f"  ⚠️  ATTACK DETECTED! Class: {label}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No flows captured")
                    
        except KeyboardInterrupt:
            print("\n\nStopping capture early...")
        
        self.capturing = False
        
        # Print final report
        print("\n")
        print("=" * 60)
        print("                    FINAL REPORT")
        print("=" * 60)
        total_time = time.time() - start_time
        print(f"Total capture time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Total records (3-sec intervals): {total_records}")
        print(f"Total flows captured: {total_flows}")
        print("-" * 60)
        print(f"  ✓ Benign intervals: {total_benign} ({100*total_benign/max(total_records,1):.1f}%)")
        print(f"  ⚠️  Attack intervals: {total_attacks} ({100*total_attacks/max(total_records,1):.1f}%)")
        
        if attack_classes:
            print("\n  Attack breakdown by class:")
            for cls, count in sorted(attack_classes.items()):
                print(f"    Class {cls}: {count} ({100*count/max(total_attacks,1):.1f}% of attacks)")
        
        print("=" * 60)
        print("Capture complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Real-time network anomaly detection')
    parser.add_argument('-i', '--interface', default=None, help='Network interface to capture on')
    parser.add_argument('-t', '--interval', type=int, default=3, help='Capture interval in seconds')
    parser.add_argument('-d', '--duration', type=int, default=300, help='Total capture duration in seconds (default: 300 = 5 min)')
    args = parser.parse_args()
    
    # Default paths - matching actual files in models directory
    model_path = 'models/vae_full_model .pth'  # Note: space in filename
    column_mapping_path = 'models/column_min_max_mapping.csv'
    catboost_path = 'models/catboost_model.json'
    top_features_path = 'models/top_features.joblib'
    
    detector = RealTimeDetector(
        model_path=model_path,
        column_mapping_path=column_mapping_path,
        catboost_path=catboost_path,
        top_features_path=top_features_path
    )
    detector.run(interface=args.interface, interval=args.interval, duration=args.duration)


if __name__ == '__main__':
    main()
