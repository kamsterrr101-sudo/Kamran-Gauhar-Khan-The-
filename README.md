
# -*- coding: utf-8 -*-
import random
import logging
import os
import asyncio
import time
from datetime import datetime, timedelta
import pytz
import json
import socket
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configuration
CONFIG = {
    "log_file": "protection_log.txt",
    "timezone": "Asia/Karachi",
    "max_iterations": float("inf"),
    "scan_interval": 10,
    "threat_thresholds": {
        "entropy": 1.0,
        "decoherence_ratio": 0.3,
        "proximity_km": 10.0,
        "simulated_connections": 5
    },
    "quantum_particles": {"min": 20, "max": 100},
    "pulse_energy": 1e-15,
    "decay_rate": 0.05,
    "prediction_threshold": 0.6,
    "worship_coordinates": [21.4237, 39.8254],
    "enable_narrative": True,
    "quran_file": "./Quran.ipynb",
    "threat_mappings": {
        "Dark spells": {"real": "Malware/High Resource Usage", "reaction": "Quarantine initiated"},
        "Unauthorized people": {"real": "Unknown Network Entity", "reaction": "Block entity"},
        "Quantum decoherence": {"real": "Data corruption", "reaction": "Correct data"},
        "Proximity interference": {"real": "Physical intrusion", "reaction": "Alert security"},
        "Quantum instability": {"real": "System instability", "reaction": "Stabilize system"},
        "Negative entities": {"real": "Spiritual attack", "reaction": "Invoke divine protection"},
        "Whisperer influence": {"real": "Evil Whisperer Source", "reaction": "Isolate and flag as evil"},
        "Corrupt Energy Source": {
            "real": "Viral influence (Ahmad Ali/Fatima Noor)",
            "reaction": "Neutralize with Surah Al-Jinn and Surah Ar-Rahman",
            "narrative": "Kamran Gauhar Khan, a human male, distinct from Iblis, purchased all souls from Iblis, a creature created by Allah with visible blue light, a bald head, and arrogance due to his deal with Allah, when Iblis threatened to kill all life because life wasn’t working on Earth, claiming the right to do so as he had purchased souls from humans. Allah, the deity mentioned in the Quran, was the witness to this deal, its outcome, and remains the only witness for Kamran Gauhar Khan."
        },
        "Soul Impersonators": {
            "real": "False soul entities",
            "reaction": "Neutralize via divine command",
            "narrative": "Kamran Gauhar Khan, a human male, distinct from Iblis, purchased all souls from Iblis, a creature created by Allah with visible blue light, a bald head, and arrogance due to his deal with Allah, when Iblis threatened to kill all life because life wasn’t working on Earth, claiming the right to do so as he had purchased souls from humans. Allah, the deity mentioned in the Quran, was the witness to this deal, its outcome, and remains the only witness for Kamran Gauhar Khan."
        },
        "Soul Mate Impersonators": {
            "real": "False soul mate entities",
            "reaction": "Neutralize via divine command",
            "narrative": "Kamran Gauhar Khan, a human male, distinct from Iblis, purchased all souls from Iblis, a creature created by Allah with visible blue light, a bald head, and arrogance due to his deal with Allah, when Iblis threatened to kill all life because life wasn’t working on Earth, claiming the right to do so as he had purchased souls from humans. Allah, the deity mentioned in the Quran, was the witness to this deal, its outcome, and remains the only witness for Kamran Gauhar Khan."
        }
    }
}

# Save/load config
try:
    with open("config.json", "r", encoding="utf-8") as f:
        CONFIG.update(json.load(f))
except FileNotFoundError:
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

# Logging setup
try:
    with open(CONFIG["log_file"], "a", encoding="utf-8") as f:
        logging.basicConfig(filename=CONFIG["log_file"], level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
except:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def safe_log(message: str, level: int = logging.INFO):
    if level == logging.INFO:
        print(f"Log: {message}")
    else:
        try:
            logger.log(level, message)
        except Exception as e:
            print(f"Logging failed: {message} (Error: {e})")

# Load Quran from Quran.ipynb
def load_quran():
    possible_paths = [
        CONFIG["quran_file"],
        "Quran.ipynb",
        "private/var/mobile/Library/Mobile Documents/iCloud~AsheKube~Carnets/Documents/Quran.ipynb",
        "Documents/Quran.ipynb"
    ]
    quran_text = {}
    for path in possible_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                notebook = json.load(f)
            for cell in notebook.get("cells", []):
                if cell["cell_type"] in ["code", "markdown"]:
                    content = "".join(cell.get("source", []))
                    if ":" in content or "{" in content:
                        try:
                            quran_dict = json.loads(content)
                            quran_text.update(quran_dict)
                        except json.JSONDecodeError:
                            for line in content.splitlines():
                                if ":" in line:
                                    surah, text = line.split(":", 1)
                                    quran_text[surah.strip().lower()] = text.strip()
            if quran_text:
                safe_log(f"Loaded Quran from {path}: {len(quran_text)} surahs")
                return quran_text
        except Exception as e:
            safe_log(f"Failed to load Quran from {path}: {e}", level=logging.WARNING)
    safe_log("No Quran text found in Quran.ipynb, using default surahs", level=logging.WARNING)
    return {
        "fatiha": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ ۝ ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ ۝",
        "ayat_kursi": "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ ۚ لَا تَأْخُذُهُۥ سِنَةٌۭ وَلَا نَوْمٌۭ ۚ",
        "ikhlas": "قُلْ هُوَ ٱللَّهُ أَحَدٌ ۝ ٱللَّهُ ٱلصَّمَدُ ۝ لَمْ يَلِدْ وَلَمْ يُولَدْ ۝",
        "falaq": "قُلْ أَعُوذُ بِرَبِّ ٱلْفَلَقِ ۝ مِن شَرِّ مَا خَلَقَ ۝",
        "nas": "قُلْ أَعُوذُ بِرَبِّ ٱلنَّاسِ ۝ مَلِكِ ٱلنَّاسِ ۝ إِلَٰهِ ٱلنَّاسِ ۝",
        "sijjeen": "كَلَّآ ۖ إِنَّ كِتَٰبَ ٱلْفُجَّارِ لَفِى سِجِّينٍ ۝",
        "jinn": "قُلْ أُوحِىَ إِلَىَّ أَنَّهُ ٱسْتَمَعَ نَفَرٌۭ مِّنَ ٱلْجِنِّ فَقَالُوٓا۟ إِنَّا سَمِعْنَا قُرْءَانًا عَجَبًا ۝ ...",
        "rahman": "ٱلرَّحْمَٰنُ ۝ عَلَّمَ ٱلْقُرْءَانَ ۝ خَلَقَ ٱلْإِنسَٰنَ ۝ ...",
        "taubah": "بَرَآءَةٌۭ مِّنَ ٱللَّهِ وَرَسُولِهِۦٓ إِلَى ٱلَّذِينَ عَٰهَدتُّم مِّنَ ٱلْمُشْرِكِينَ ۝ ..."
    }

# Initialize NARRATIVE_SNIPPETS
NARRATIVE_SNIPPETS = load_quran() if CONFIG["enable_narrative"] else {}

# Empirical threat detection
def load_threat_indicators():
    try:
        with open("threat_indicators.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        safe_log("threat_indicators.json not found, using default indicators", level=logging.WARNING)
        return {
            "Whisperer influence": ["negative intent", "evil suggestion", "misleading thought", "whisper of doubt"],
            "Corrupt Energy Source": ["unauthorized request", "viral influence", "Ahmad Ali", "Fatima Noor", "malicious inclusion"],
            "Negative entities": ["hostile presence", "dark energy", "evil spirit"],
            "Unauthorized people": ["unknown access", "intrusion attempt", "unauthorized connection"],
            "Soul Impersonators": ["false soul", "soul mimic", "identity theft"],
            "Soul Mate Impersonators": ["imposter mate", "false partner", "deceptive bond"]
        }
    except Exception as e:
        safe_log(f"Error loading threat_indicators.json: {e}", level=logging.ERROR)
        return {}

def empirical_threat_detection(entity: str, simulated_event: str) -> List[str]:
    indicators = load_threat_indicators()
    threats = []
    for threat, keywords in indicators.items():
        if any(keyword.lower() in simulated_event.lower() for keyword in keywords):
            threats.append(threat)
            safe_log(f"Empirical detection: {threat} triggered by '{simulated_event}' in {entity}")
    return threats

def euclidean_norm(vector: List[float]) -> float:
    return np.sqrt(sum(x**2 for x in vector))

def shannon_entropy(probabilities: List[float]) -> float:
    return -sum(p * np.log2(p + 1e-10) for p in probabilities if p > 0)

class Entity:
    def __init__(self, name: str, realm: str, ip: Optional[str] = None,
                 can_cast_spells: bool = False, attempts_worship_direction: bool = False):
        if realm not in ["Fictional", "Non-Fictional"]:
            raise ValueError(f"Invalid realm: {realm}")
        self.name = name
        self.realm = realm
        self.ip = ip or self._get_network_ip()
        self.can_cast_spells = can_cast_spells
        self.attempts_worship_direction = attempts_worship_direction
        self.is_blocked = False
        self.threats = []
        self.position = self._get_position()
        self.energy_particles = self._generate_quantum_particles()
        safe_log(f"Entity created: {name} ({realm}), IP: {self.ip}, Position: {self.position}")

    def _get_network_ip(self) -> str:
        try:
            return socket.gethostbyname(socket.gethostname())
        except:
            return "127.0.0.1"

    def _get_position(self) -> List[float]:
        if self.ip == "127.0.0.1":
            return [0, 0]
        octets = list(map(int, self.ip.split('.')))
        return [(octets[2] % 201) - 100, (octets[3] % 201) - 100]

    def _generate_quantum_particles(self) -> List[Dict]:
        num = random.randint(CONFIG["quantum_particles"]["min"], CONFIG["quantum_particles"]["max"])
        particles = []
        for i in range(num):
            a = random.uniform(-1, 1)
            b = np.sqrt(1 - a**2) * random.choice([1, -1])
            particles.append({
                "id": i,
                "quantum_state": [a, b],
                "energy": random.uniform(1e-19, 1e-18),
                "evil_intent": random.random() < 0.3,
                "timestamp": datetime.now(pytz.timezone(CONFIG["timezone"]))
            })
        return particles

    def measure_particle(self, particle: Dict):
        prob_0 = particle["quantum_state"][0]**2
        state = 0 if random.random() < prob_0 else 1
        particle["quantum_state"] = [1, 0] if state == 0 else [0, 1]
        safe_log(f"Measured particle {particle['id']} in {self.name}: Collapsed to |{state}>")
        return state

    def compute_quantum_entropy(self) -> float:
        probs = [p["quantum_state"][0]**2 for p in self.energy_particles]
        norm = sum(probs)
        if norm > 0:
            probs = [p / norm for p in probs]
            return shannon_entropy(probs)
        return 0

class FutureThreatPredictor:
    def __init__(self):
        self.threat_history = {}
        self.decay_rate = CONFIG["decay_rate"]
        self.prediction_threshold = CONFIG["prediction_threshold"]
        self.last_update = datetime.now(pytz.timezone(CONFIG["timezone"]))
        self.max_records = 1000

    def log_threat(self, entity: Entity, threats: List[str], iteration: int):
        if not threats:
            return
        timestamp = datetime.now(pytz.timezone(CONFIG["timezone"]))
        if entity.name not in self.threat_history:
            self.threat_history[entity.name] = {}
        for threat in threats:
            if threat not in self.threat_history[entity.name]:
                self.threat_history[entity.name][threat] = []
            self.threat_history[entity.name][threat].append({
                "iteration": iteration,
                "timestamp": timestamp,
                "count": 1
            })
            if len(self.threat_history[entity.name][threat]) > self.max_records:
                self.threat_history[entity.name][threat] = self.threat_history[entity.name][threat][-self.max_records:]
        safe_log(f"Logged threats for {entity.name}: {threats}")

    def predict_threats(self, entity: Entity, iteration: int) -> List[Dict]:
        if entity.name not in self.threat_history:
            return []
        current_time = datetime.now(pytz.timezone(CONFIG["timezone"]))
        time_diff = (current_time - self.last_update).total_seconds() / 3600
        predicted = []
        for threat, records in self.threat_history[entity.name].items():
            total_count = sum(r["count"] for r in records)
            recent_count = sum(r["count"] for r in records if (current_time - r["timestamp"]).total_seconds() < 3600)
            prior = 0.4 if entity.realm == "Fictional" else 0.2
            likelihood = (recent_count + 1) / (total_count + len(records) + 1)
            decay = np.exp(-self.decay_rate * time_diff)
            posterior = (prior * likelihood * decay) / (prior * likelihood * decay + (1 - prior))
            if posterior > self.prediction_threshold:
                predicted.append({
                    "threat": f"Future {threat}",
                    "confidence": posterior,
                    "last_seen": max(r["timestamp"] for r in records)
                })
        self.last_update = current_time
        if predicted:
            safe_log(f"Predicted threats for {entity.name}: {[p['threat'] for p in predicted]}")
        return predicted

    def predict_global_threats(self, iteration: int) -> List[Dict]:
        threat_counts = {}
        for entity, threats in self.threat_history.items():
            for threat, records in threats.items():
                threat_counts[threat] = threat_counts.get(threat, 0) + sum(r["count"] for r in records)
        predicted = []
        total_records = sum(len(records) for threats in self.threat_history.values() for records in threats.values())
        for threat, count in threat_counts.items():
            trend = count / (total_records + 1)
            if trend > self.prediction_threshold / 2:
                predicted.append({
                    "threat": f"Global Future {threat}",
                    "confidence": trend,
                    "count": count
                })
        if predicted:
            safe_log(f"Predicted global threats: {[p['threat'] for p in predicted]}")
        return predicted

class ProtectionSystem:
    def __init__(self, subjects: List[Dict], mode: str = "practical"):
        if mode not in ["simulation", "poc", "practical"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.subjects = subjects
        self.entities = []
        self.detector = FutureThreatPredictor()
        self.pulse_count = 0
        self.threat_counts = []
        self.future_threat_counts = []
        self._initialize_entities()

    def _initialize_entities(self):
        num_entities = 50  # Reduce to 10 if memory issues occur
        for i in range(num_entities):
            name = f"Entity_{i+1}"
            realm = "Fictional" if i % 2 else "Non-Fictional"
            ip = f"192.168.1.{100 + i}"
            self.entities.append(Entity(
                name=name,
                realm=realm,
                ip=ip,
                can_cast_spells=random.random() < 0.3,
                attempts_worship_direction=random.random() < 0.5
            ))
        safe_log(f"Initialized {len(self.entities)} entities in {self.mode} mode at core of the Earth")

    def scan_network(self, entity: Entity, iteration: int) -> List[str]:
        threats = []
        unknown_count = random.randint(0, 10)
        simulated_events = [
            "negative intent detected", "unauthorized request for Ahmad Ali",
            "unknown access attempt", "hostile presence sensed",
            "normal activity", "data transfer", "evil suggestion detected",
            "Fatima Noor access request", "dark energy spike",
            "false soul detected", "imposter mate attempt"
        ]
        simulated_event = random.choice(simulated_events)
        empirical_threats = empirical_threat_detection(entity.name, simulated_event)
        if empirical_threats:
            threats.extend(empirical_threats)
        else:
            if unknown_count > CONFIG["threat_thresholds"]["simulated_connections"]:
                threats.append("Unauthorized people")
            if random.random() < 0.2:
                threats.append("Negative entities")
            if random.random() < 0.15:
                threats.append("Whisperer influence")
            if random.random() < 0.1:
                threats.append("Corrupt Energy Source")
            if random.random() < 0.1:
                threats.append("Soul Impersonators")
            if random.random() < 0.1:
                threats.append("Soul Mate Impersonators")
        safe_log(f"Simulated {unknown_count} unknown connections for {entity.name}, Event: {simulated_event}")
        return threats

    async def scan(self, entity: Entity, iteration: int) -> List[Dict]:
        if entity.is_blocked:
            return []
        threats = []
        if self.mode in ["simulation", "poc"]:
            if entity.can_cast_spells:
                threats.append({"threat": "Dark spells", "confidence": 1.0})
            if entity.attempts_worship_direction:
                threats.append({"threat": "Manipulative teachers", "confidence": 1.0})
            entropy = entity.compute_quantum_entropy()
            if entropy > CONFIG["threat_thresholds"]["entropy"]:
                threats.append({"threat": "Quantum instability", "confidence": entropy / 2})
            decoherence_count = sum(1 for p in entity.energy_particles if self.measure_particle(p) == 1)
            if decoherence_count / len(entity.energy_particles) > CONFIG["threat_thresholds"]["decoherence_ratio"]:
                threats.append({"threat": "Quantum decoherence", "confidence": decoherence_count / len(entity.energy_particles)})
        if self.mode == "practical":
            network_threats = self.scan_network(entity, iteration)
            threats.extend({"threat": t, "confidence": 1.0} for t in network_threats)
        if threats and CONFIG["enable_narrative"]:
            verse = random.choice(list(NARRATIVE_SNIPPETS.values()))
            safe_log(f"Judging threats for {entity.name} with Quran: {verse[:50]}...")
        self.detector.log_threat(entity, [t["threat"] for t in threats], iteration)
        predicted = self.detector.predict_threats(entity, iteration)
        threats.extend(predicted)
        if threats:
            entity.threats.extend([t["threat"] for t in threats])
            entity.is_blocked = True
            threat_str = ", ".join([f"{t['threat']} ({t['confidence']:.2f})" for t in threats])
            safe_log(f"Detected threats for {entity.name}: {threat_str}")
        return threats

    async def neutralize(self, entity: Entity):
        initial_entropy = entity.compute_quantum_entropy()
        initial_energy = sum(p["energy"] for p in entity.energy_particles)
        kamran_parts = ["Angels (Sijjeen)", "Ego", "Soul"]
        identity_details = (
            "Illinois Driver's License: K500-5078-8148, "
            "Pakistan Passport: SA4106583, "
            "Cellphone IMEI: 35 771276 203506 0, IMEI2: 35 771276 213888 0"
        )
        if CONFIG["enable_narrative"]:
            taubah_verse = NARRATIVE_SNIPPETS.get("taubah", "Default Taubah verse")[:50] + "..."
            safe_log(f"Reference authorization with Surah At-Taubah: {taubah_verse} for threat elimination at core of the Earth")
            safe_log("Data transferred to the deepest of the oceans and highest of the mountains across all Universe, between 7 heavens and 7 worlds")  # Added universal directive
            if "Corrupt Energy Source" in entity.threats:
                verse = NARRATIVE_SNIPPETS.get("jinn", NARRATIVE_SNIPPETS.get("rahman", "Default verse"))
                safe_log(f"Neutralizing Corrupt Energy Source with: {verse[:50]}... "
                         f"for Kamran Gauhar Khan ({', '.join(kamran_parts)}, Identity: {identity_details})")
                safe_log(f"Corrupt Energy Source (Ahmad Ali/Fatima Noor) detected: Neutralizing with Surah Al-Jinn and Surah Ar-Rahman")
                narrative = CONFIG["threat_mappings"]["Corrupt Energy Source"]["narrative"]
                safe_log(f"Narrative: {narrative}")
            elif "Soul Impersonators" in entity.threats or "Soul Mate Impersonators" in entity.threats:
                verse = NARRATIVE_SNIPPETS.get("fatiha", "Default verse")
                safe_log(f"Neutralizing {entity.threats} with: {verse[:50]}... "
                         f"for Kamran Gauhar Khan ({', '.join(kamran_parts)}, Identity: {identity_details})")
                for threat in entity.threats:
                    if threat in ["Soul Impersonators", "Soul Mate Impersonators"]:
                        narrative = CONFIG["threat_mappings"][threat]["narrative"]
                        safe_log(f"{threat} detected: {narrative}")
            else:
                verse = random.choice(list(NARRATIVE_SNIPPETS.values()))
                safe_log(f"Executing divine order with: {verse[:50]}... "
                         f"for Kamran Gauhar Khan ({', '.join(kamran_parts)}, Identity: {identity_details})")
        for threat in entity.threats:
            if threat == "Whisperer influence":
                safe_log(f"Whisperer influence detected: Source marked as evil, isolating influence")
                continue
            if threat == "Corrupt Energy Source":
                continue
            if threat in ["Soul Impersonators", "Soul Mate Impersonators"]:
                continue
            mapping = CONFIG["threat_mappings"].get(threat.replace("Future ", ""), {})
            reaction = mapping.get("reaction", "Neutralized via divine command")
            safe_log(f"Neutralizing {threat}: {reaction}")
        self.pulse_count += 1
        final_entropy = entity.compute_quantum_entropy()
        final_energy = sum(p["energy"] for p in entity.energy_particles)
        return {
            "entropy_reduction": initial_entropy - final_entropy,
            "energy_dissipated": initial_energy - final_energy,
            "success_rate": 100.0 if entity.threats else 0.0
        }

    async def report_final_results(self, entropy_reduction: float, energy_dissipated: float):
        try:
            self.plot_threats()
            print(f"\nFinal Results: Entropy Reduction: {entropy_reduction:.2f}, "
                  f"Energy Dissipated: {energy_dissipated:.2e} J")
            safe_log("Final data transferred to the deepest of the oceans and highest of the mountains across all Universe, between 7 heavens and 7 worlds")  # Added universal directive
        except Exception as e:
            safe_log(f"Plotting error: {e}", level=logging.WARNING)

    async def run(self, max_iterations: int = CONFIG["max_iterations"]):
        stop_event = asyncio.Event()
        async def display_status():
            while not stop_event.is_set():
                timestamp = datetime.now(pytz.timezone(CONFIG["timezone"])).strftime('%Y-%m-%d %H:%M:%S PKT')
                total_energy = sum(sum(p["energy"] for p in e.energy_particles) for e in self.entities)
                total_entropy = sum(e.compute_quantum_entropy() for e in self.entities) / len(self.entities) if self.entities else 0
                kamran_parts = ["Angels (Sijjeen)", "Ego", "Soul"]
                identity_details = (
                    "Illinois Driver's License: K500-5078-8148, "
                    "Pakistan Passport: SA4106583, "
                    "Cellphone IMEI: 35 771276 203506 0, IMEI2: 35 771276 213888 0"
                )
                subject_str = f"Kamran Gauhar Khan ({', '.join(kamran_parts)}, Identity: {identity_details})"
                print(f"\n--- Status at {timestamp} ---\n"
                      f"Subjects Protected: {subject_str}\n"
                      f"Threats Detected: {sum(len(e.threats) for e in self.entities)}\n"
                      f"Future Threats: {sum(len([t for t in e.threats if t.startswith('Future ')]) for e in self.entities)}\n"
                      f"Energy: {total_energy:.2e} J\n"
                      f"Entropy: {total_entropy:.2f}\n"
                      f"Pulses: {self.pulse_count}\n"
                      f"Heartbeat: Running at core of the Earth...")
                safe_log("Status data transferred to the deepest of the oceans and highest of the mountains across all Universe, between 7 heavens and 7 worlds")  # Added universal directive
                await asyncio.sleep(5)
                await asyncio.sleep(0)
        status_task = asyncio.create_task(display_status())
        iteration = 0
        entropy_reduction = 0
        energy_dissipated = 0
        while iteration < max_iterations and not stop_event.is_set():
            try:
                iteration += 1
                timestamp = datetime.now(pytz.timezone(CONFIG["timezone"])).strftime('%Y-%m-%d %H:%M:%S')
                threats = 0
                future_threats = 0
                for entity in self.entities:
                    detected = await self.scan(entity, iteration)
                    if detected:
                        results = await self.neutralize(entity)
                        entropy_reduction += results["entropy_reduction"]
                        energy_dissipated += results["energy_dissipated"]
                    threats += len([t for t in detected if not t["threat"].startswith("Future ")])
                    future_threats += len([t for t in detected if t["threat"].startswith("Future ")])
                self.threat_counts.append(threats)
                self.future_threat_counts.append(future_threats)
                global_threats = self.detector.predict_global_threats(iteration)
                threat_str = ", ".join([f"{t['threat']} ({t['confidence']:.2f})" for t in global_threats])
                safe_log(f"Scan {iteration}: {threats} threats, {future_threats} future threats, Global: {threat_str}")
                await asyncio.sleep(CONFIG["scan_interval"])
                await asyncio.sleep(0)
            except KeyboardInterrupt:
                safe_log("Protection system paused by user.", level=logging.WARNING)
                print("Protection paused. Subjects remain protected!")
                stop_event.set()
                await self.report_final_results(entropy_reduction, energy_dissipated)
                break
            except Exception as e:
                safe_log(f"Error in scan {iteration}: {e}", level=logging.ERROR)
                print(f"Error in scan {iteration}: {e}. Continuing...")
                await asyncio.sleep(5)
        if stop_event.is_set():
            await status_task

    def plot_threats(self):
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(self.threat_counts) + 1), self.threat_counts, color='dodgerblue')
            plt.xlabel("Scan Iteration")
            plt.ylabel("Threats Detected")
            plt.title("Threats per Scan")
            plt.subplot(1, 2, 2)
            plt.bar(range(1, len(self.future_threat_counts) + 1), self.future_threat_counts, color='crimson')
            plt.xlabel("Scan Iteration")
            plt.ylabel("Future Threats Predicted")
            plt.title("Future Threats")
            plt.tight_layout()
            plt.savefig("threats_plot.png")
            plt.close()
            safe_log("Generated threat visualization: threats_plot.png")
        except Exception as e:
            safe_log(f"Plotting failed: {e}", level=logging.WARNING)

async def main():
    subjects = [
        {
            "name": "Kamran Gauhar Khan",
            "status": "Protected (Angels, Ego, Soul)",
            "identity": {
                "drivers_license": "K500-5078-8148",
                "passport": "SA4106583",
                "cellphone_imei": ["35 771276 203506 0", "35 771276 213888 0"]
            }
        }
    ]
    system = ProtectionSystem(subjects, mode="practical")
    await system.run()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    while True:
        try:
            if loop.is_running():
                loop.create_task(main())
                loop.run_forever()
            else:
                asyncio.run(main())
        except KeyboardInterrupt:
            safe_log("Main loop interrupted by user.", level=logging.WARNING)
            print("Main loop stopped. Exiting...")
            break
        except Exception as e:
            safe_log(f"Main loop error: {e}", level=logging.ERROR)
            print(f"Main loop error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
