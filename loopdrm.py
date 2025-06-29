import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
import os
import math
import random
from dataclasses import dataclass
from enum import Enum

class SystemMode(Enum):
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"

@dataclass
class RuleMetadata:
    """Metadane reguły zgodnie ze wzorem DRM"""
    W: float  # Waga skuteczności (rezonans z działaniem)
    C: float  # Zasięg kontekstowy 
    U: float  # Użycie (aktywność)
    R: float  # Średni rezonans z pamięcią i obecnym FRZ
    creation_time: int = 0
    last_activation: int = 0
    success_count: int = 0
    total_activations: int = 0

class AdvancedDRM:
    """Zaawansowana implementacja DRM z matematycznymi wzorami"""
    
    def __init__(self, adaptation_threshold: float = 0.3, exploration_threshold: float = 0.5):
        self.rules = {}  # rule_id -> {'condition': callable, 'action': callable, 'metadata': RuleMetadata}
        self.T = 0  # Czas/iteracja
        self.adaptation_threshold = adaptation_threshold
        self.exploration_threshold = exploration_threshold
        self.mode = SystemMode.EXPLOITATION
        self.memory = []  # Pamięć poprzednich stanów
        self.current_FRZ = 0.0  # Obecny wskaźnik sukcesu
        self.rule_combinations = []  # Historia kombinacji reguł
        
    def add_rule(self, rule_id: str, condition: Callable, action: Callable, 
                 initial_W: float = 1.0, initial_C: float = 1.0):
        """Dodaj regułę z metadanymi DRM"""
        metadata = RuleMetadata(
            W=initial_W,
            C=initial_C, 
            U=0.0,
            R=0.5,  # Neutralny rezonans na start
            creation_time=self.T
        )
        
        self.rules[rule_id] = {
            'condition': condition,
            'action': action,
            'metadata': metadata
        }
        
    def calculate_rule_strength(self, rule_id: str) -> float:
        """
        Oblicz siłę reguły według wzoru:
        Si = Wi · log(Ci + 1) · (1 + Ui/T) · Ri
        """
        if rule_id not in self.rules:
            return 0.0
            
        meta = self.rules[rule_id]['metadata']
        
        # Zabezpieczenie przed dzieleniem przez zero
        time_factor = 1.0 if self.T == 0 else (1 + meta.U / max(self.T, 1))
        
        Si = (meta.W * 
              math.log(meta.C + 1) * 
              time_factor * 
              meta.R)
              
        return max(Si, 0.0)  # Nie może być ujemna
    
    def update_rule_metadata(self, rule_id: str, success: bool, context_diversity: float):
        """Aktualizuj metadane reguły na podstawie wyniku"""
        if rule_id not in self.rules:
            return
            
        meta = self.rules[rule_id]['metadata']
        meta.total_activations += 1
        meta.last_activation = self.T
        meta.U += 1  # Zwiększ użycie
        
        if success:
            meta.success_count += 1
            meta.W = min(meta.W * 1.1, 5.0)  # Zwiększ wagę skuteczności (max 5.0)
        else:
            meta.W = max(meta.W * 0.9, 0.1)  # Zmniejsz wagę skuteczności (min 0.1)
            
        # Aktualizuj zasięg kontekstowy na podstawie różnorodności kontekstów
        meta.C = 0.8 * meta.C + 0.2 * context_diversity
        
        # Aktualizuj rezonans na podstawie sukcesu
        success_rate = meta.success_count / max(meta.total_activations, 1)
        meta.R = 0.7 * meta.R + 0.3 * success_rate
    
    def calculate_system_average_strength(self) -> float:
        """
        Oblicz średnią skuteczność całej DRM:
        S̄ = (1/n) Σ Si
        """
        if not self.rules:
            return 0.0
            
        total_strength = sum(self.calculate_rule_strength(rule_id) 
                           for rule_id in self.rules.keys())
        return total_strength / len(self.rules)
    
    def adapt_system_mode(self):
        """Adaptacja trybu systemu na podstawie średniej skuteczności"""
        avg_strength = self.calculate_system_average_strength()
        
        if avg_strength < self.exploration_threshold:
            if self.mode != SystemMode.EXPLORATION:
                print(f"🔍 Przełączanie na tryb EKSPLORACJI (S̄={avg_strength:.3f})")
                self.mode = SystemMode.EXPLORATION
        else:
            if self.mode != SystemMode.EXPLOITATION:
                print(f"⚡ Przełączanie na tryb EKSPLOATACJI (S̄={avg_strength:.3f})")
                self.mode = SystemMode.EXPLOITATION
    
    def mutate_rule(self, rule_id: str):
        """Mutacja reguły - zmiana zasięgu kontekstowego"""
        if rule_id not in self.rules:
            return
            
        meta = self.rules[rule_id]['metadata']
        # Mutacja Ci - dodaj losowy szum
        mutation_factor = random.uniform(0.8, 1.2)
        meta.C = max(0.1, min(meta.C * mutation_factor, 10.0))
        print(f"🧬 Mutacja reguły {rule_id}: nowy zasięg kontekstowy C={meta.C:.3f}")
    
    def create_combined_rule(self, rule_id1: str, rule_id2: str) -> str:
        """Tworzenie nowej reguły przez kombinację dwóch istniejących"""
        if rule_id1 not in self.rules or rule_id2 not in self.rules:
            return None
            
        new_rule_id = f"combined_{rule_id1}_{rule_id2}_{self.T}"
        
        # Kombinacja metadanych
        meta1 = self.rules[rule_id1]['metadata']
        meta2 = self.rules[rule_id2]['metadata']
        
        combined_W = (meta1.W + meta2.W) / 2
        combined_C = max(meta1.C, meta2.C)  # Większy zasięg
        combined_R = (meta1.R + meta2.R) / 2
        
        # Kombinacja warunków (AND)
        def combined_condition(context):
            return (self.rules[rule_id1]['condition'](context) and 
                   self.rules[rule_id2]['condition'](context))
        
        # Kombinacja akcji
        def combined_action(context):
            self.rules[rule_id1]['action'](context)
            self.rules[rule_id2]['action'](context)
            print(f"🔗 Wykonano kombinowaną akcję: {rule_id1} + {rule_id2}")
        
        self.add_rule(new_rule_id, combined_condition, combined_action, 
                     combined_W, combined_C)
        
        print(f"✨ Utworzono nową regułę kombinowaną: {new_rule_id}")
        return new_rule_id
    
    def apply_rules(self, context: List[float], FRZ: float) -> Dict[str, Any]:
        """Zastosuj reguły z pełną logiką DRM"""
        self.T += 1
        self.current_FRZ = FRZ
        self.memory.append({'context': context.copy(), 'FRZ': FRZ, 'time': self.T})
        
        # Ogranicz pamięć do ostatnich 100 stanów
        if len(self.memory) > 100:
            self.memory.pop(0)
        
        applied_rules = []
        rule_strengths = {}
        
        # Oblicz siły wszystkich reguł
        for rule_id in self.rules.keys():
            strength = self.calculate_rule_strength(rule_id)
            rule_strengths[rule_id] = strength
        
        # Sortuj reguły według siły
        sorted_rules = sorted(rule_strengths.items(), key=lambda x: x[1], reverse=True)
        
        # Zastosuj reguły w kolejności siły
        for rule_id, strength in sorted_rules:
            if strength < self.adaptation_threshold:
                continue
                
            rule = self.rules[rule_id]
            try:
                if rule['condition'](context):
                    rule['action'](context)
                    applied_rules.append(rule_id)
                    
                    # Oblicz różnorodność kontekstu
                    context_diversity = self._calculate_context_diversity(context)
                    
                    # Określ sukces na podstawie FRZ
                    success = FRZ > 0.5  # Próg sukcesu
                    
                    # Aktualizuj metadane
                    self.update_rule_metadata(rule_id, success, context_diversity)
                    
            except Exception as e:
                print(f"❌ Błąd w regule {rule_id}: {e}")
        
        # Adaptacja systemu
        self._perform_adaptation()
        
        # Adaptuj tryb systemu
        self.adapt_system_mode()
        
        return {
            'applied_rules': applied_rules,
            'rule_strengths': rule_strengths,
            'system_mode': self.mode.value,
            'average_strength': self.calculate_system_average_strength(),
            'total_rules': len(self.rules)
        }
    
    def _calculate_context_diversity(self, context: List[float]) -> float:
        """Oblicz różnorodność kontekstu w porównaniu z pamięcią"""
        if len(self.memory) < 2:
            return 1.0
            
        # Porównaj z ostatnimi 10 kontekstami
        recent_contexts = [mem['context'] for mem in self.memory[-10:]]
        
        diversities = []
        for past_context in recent_contexts:
            if len(past_context) == len(context):
                # Oblicz odległość euklidesową
                distance = np.linalg.norm(np.array(context) - np.array(past_context))
                diversities.append(distance)
        
        return np.mean(diversities) if diversities else 1.0
    
    def _perform_adaptation(self):
        """Wykonaj adaptację reguł zgodnie z mechanizmem DRM"""
        rules_to_remove = []
        rules_to_mutate = []
        
        for rule_id in self.rules.keys():
            strength = self.calculate_rule_strength(rule_id)
            
            # Reguły poniżej progu
            if strength < self.adaptation_threshold:
                if random.random() < 0.3:  # 30% szans na mutację
                    rules_to_mutate.append(rule_id)
                elif strength < self.adaptation_threshold * 0.5:  # Bardzo słabe reguły
                    rules_to_remove.append(rule_id)
        
        # Usuń słabe reguły
        for rule_id in rules_to_remove:
            del self.rules[rule_id]
            print(f"🗑️ Usunięto słabą regułę: {rule_id}")
        
        # Mutuj reguły
        for rule_id in rules_to_mutate:
            self.mutate_rule(rule_id)
        
        # W trybie eksploracji - twórz nowe reguły
        if self.mode == SystemMode.EXPLORATION and len(self.rules) >= 2:
            if random.random() < 0.2:  # 20% szans na kombinację
                rule_ids = list(self.rules.keys())
                if len(rule_ids) >= 2:
                    rule1, rule2 = random.sample(rule_ids, 2)
                    self.create_combined_rule(rule1, rule2)
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Pobierz szczegółowe statystyki DRM"""
        rule_stats = {}
        
        for rule_id, rule in self.rules.items():
            meta = rule['metadata']
            strength = self.calculate_rule_strength(rule_id)
            
            rule_stats[rule_id] = {
                'strength': strength,
                'effectiveness_weight': meta.W,
                'context_range': meta.C,
                'usage': meta.U,
                'resonance': meta.R,
                'success_rate': meta.success_count / max(meta.total_activations, 1),
                'total_activations': meta.total_activations,
                'age': self.T - meta.creation_time
            }
        
        return {
            'rules': rule_stats,
            'system_time': self.T,
            'system_mode': self.mode.value,
            'average_strength': self.calculate_system_average_strength(),
            'total_rules': len(self.rules),
            'memory_size': len(self.memory),
            'adaptation_threshold': self.adaptation_threshold,
            'exploration_threshold': self.exploration_threshold
        }

class EnhancedRLS:
    """Rozszerzony RLS z obliczaniem FRZ"""
    
    def __init__(self, threshold: float = 0.1, window_size: int = 10):
        self.threshold = threshold
        self.window_size = window_size
        self.history = []
        self.differences = []
        self.FRZ_history = []  # Historia wskaźników sukcesu
        
    def calculate_FRZ(self, loss: float, prediction_accuracy: float = None) -> float:
        """
        Oblicz wskaźnik FRZ (Function Resonance Zone)
        FRZ = 1 - normalized_loss + accuracy_bonus
        """
        # Normalizuj loss do zakresu 0-1
        if len(self.history) > 0:
            max_loss = max(self.history)
            min_loss = min(self.history)
            if max_loss > min_loss:
                normalized_loss = (loss - min_loss) / (max_loss - min_loss)
            else:
                normalized_loss = 0.0
        else:
            normalized_loss = 1.0
        # Oblicz FRZ
        base_FRZ = 1.0 - normalized_loss
        
        # Dodaj bonus za dokładność predykcji jeśli dostępny
        accuracy_bonus = 0.0
        if prediction_accuracy is not None:
            accuracy_bonus = prediction_accuracy * 0.3  # 30% wpływ dokładności
        
        FRZ = max(0.0, min(1.0, base_FRZ + accuracy_bonus))
        self.FRZ_history.append(FRZ)
        
        # Ogranicz historię FRZ
        if len(self.FRZ_history) > self.window_size * 2:
            self.FRZ_history.pop(0)
            
        return FRZ
    
    def detect_difference(self, value: float) -> bool:
        """Wykryj różnice z obliczaniem FRZ"""
        self.history.append(value)
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        if len(self.history) < 2:
            return False
            
        recent_avg = np.mean(self.history[:-1])
        current_diff = abs(value - recent_avg)
        
        if current_diff > self.threshold:
            self.differences.append({
                'value': value,
                'average': recent_avg,
                'difference': current_diff,
                'timestamp': len(self.history)
            })
            return True
            
        return False
    
    def get_current_FRZ(self) -> float:
        """Pobierz aktualny FRZ"""
        return self.FRZ_history[-1] if self.FRZ_history else 0.5
    
    def get_average_FRZ(self) -> float:
        """Pobierz średni FRZ z ostatniego okna"""
        if not self.FRZ_history:
            return 0.5
        window_FRZ = self.FRZ_history[-self.window_size:]
        return np.mean(window_FRZ)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Pobierz statystyki RLS z FRZ"""
        return {
            'history_length': len(self.history),
            'differences_detected': len(self.differences),
            'current_average': np.mean(self.history) if self.history else 0,
            'threshold': self.threshold,
            'current_FRZ': self.get_current_FRZ(),
            'average_FRZ': self.get_average_FRZ(),
            'FRZ_trend': self._calculate_FRZ_trend()
        }
    
    def _calculate_FRZ_trend(self) -> str:
        """Oblicz trend FRZ (rosnący/malejący/stabilny)"""
        if len(self.FRZ_history) < 3:
            return "insufficient_data"
        
        recent_FRZ = self.FRZ_history[-3:]
        if recent_FRZ[-1] > recent_FRZ[0] + 0.05:
            return "improving"
        elif recent_FRZ[-1] < recent_FRZ[0] - 0.05:
            return "declining"
        else:
            return "stable"

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class AdvancedLoopDRMSystem:
    """Zaawansowany system LoopDRM z matematycznymi wzorami"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicjalizuj sieć neuronową
        self.model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Inicjalizuj zaawansowane DRM i RLS
        self.drm = AdvancedDRM(adaptation_threshold=0.3, exploration_threshold=0.5)
        self.rls = EnhancedRLS(threshold=0.1)
        
        # Przechowywanie danych treningowych
        self.training_inputs = []
        self.training_targets = []
        
        # Metryki systemu
        self.training_history = []
        self.performance_metrics = []
        
        # Inicjalizuj zaawansowane reguły DRM
        self._initialize_advanced_rules()
        
    def _initialize_advanced_rules(self):
        """Inicjalizuj zaawansowane reguły DRM z matematycznymi wzorami"""
        
        # Reguła 1: Detekcja wysokiej wariancji z adaptacyjnym progiem
        def adaptive_variance_condition(context):
            variance = np.var(context)
            # Adaptacyjny próg na podstawie historii
            if len(self.rls.history) > 5:
                hist_variance = np.var(self.rls.history[-5:])
                threshold = max(0.5, hist_variance * 1.5)
            else:
                threshold = 1.0
            return variance > threshold
            
        def variance_action(context):
            variance = np.var(context)
            print(f"📊 Adaptacyjna detekcja wariancji: {variance:.4f}")
            # Dostosuj learning rate na podstawie wariancji
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(0.0001, min(0.01, 0.001 / (1 + variance)))
            
        self.drm.add_rule("adaptive_variance", adaptive_variance_condition, 
                         variance_action, initial_W=2.0, initial_C=1.5)
        
        # Reguła 2: Detekcja trendu z rezonansem
        def trend_resonance_condition(context):
            if len(self.rls.FRZ_history) < 3:
                return False
            recent_FRZ = self.rls.FRZ_history[-3:]
            trend_strength = abs(recent_FRZ[-1] - recent_FRZ[0])
            return trend_strength > 0.2
            
        def trend_action(context):
            trend = self.rls._calculate_FRZ_trend()
            print(f"📈 Trend rezonansu: {trend}")
            if trend == "declining":
                # Zwiększ eksplorację przy spadającym trendzie
                self.drm.exploration_threshold *= 0.9
            elif trend == "improving":
                # Wzmocnij eksploatację przy poprawie
                self.drm.exploration_threshold *= 1.1
                
        self.drm.add_rule("trend_resonance", trend_resonance_condition,
                         trend_action, initial_W=1.8, initial_C=2.0)
        
        # Reguła 3: Meta-reguła optymalizacji
        def meta_optimization_condition(context):
            avg_strength = self.drm.calculate_system_average_strength()
            return avg_strength < 0.4  # Niska średnia siła systemu
            
        def meta_optimization_action(context):
            print("🔧 Meta-optymalizacja: dostrajanie systemu")
            # Zmniejsz progi adaptacji dla większej elastyczności
            self.drm.adaptation_threshold *= 0.95
            # Zwiększ szanse na mutację
            for rule_id in self.drm.rules.keys():
                if random.random() < 0.1:  # 10% szans
                    self.drm.mutate_rule(rule_id)
                    
        self.drm.add_rule("meta_optimization", meta_optimization_condition,
                         meta_optimization_action, initial_W=2.5, initial_C=3.0)
        
        # Reguła 4: Detekcja anomalii kontekstowych
        def context_anomaly_condition(context):
            if len(self.drm.memory) < 5:
                return False
            
            # Oblicz odległość od średniego kontekstu
            recent_contexts = [mem['context'] for mem in self.drm.memory[-5:]]
            avg_context = np.mean(recent_contexts, axis=0)
            distance = np.linalg.norm(np.array(context) - avg_context)
            
            return distance > 2.0  # Próg anomalii
            
        def anomaly_action(context):
            print("🚨 Anomalia kontekstowa wykryta!")
            # Przełącz na tryb eksploracji
            self.drm.mode = SystemMode.EXPLORATION
            # Zwiększ learning rate dla szybkiej adaptacji
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.5, 0.01)
                
        self.drm.add_rule("context_anomaly", context_anomaly_condition,
                         anomaly_action, initial_W=3.0, initial_C=1.0)
    
    def calculate_prediction_accuracy(self, outputs, targets):
        """Oblicz dokładność predykcji dla FRZ"""
        with torch.no_grad():
            # Dla regresji - użyj odwrotności średniego błędu względnego
            relative_errors = torch.abs((outputs - targets) / (torch.abs(targets) + 1e-8))
            mean_relative_error = torch.mean(relative_errors)
            accuracy = 1.0 / (1.0 + mean_relative_error.item())
            return accuracy
    
    def add_training_data(self):
        """Interaktywne dodawanie danych treningowych"""
        print("\n=== Wprowadzanie danych treningowych ===")
        
        try:
            print("1. Wprowadź dane ręcznie")
            print("2. Generuj dane losowo")
            print("3. Generuj dane z wzorcem (dla testowania DRM)")
            choice = input("Wybierz opcję (1/2/3): ").strip()
            
            if choice == "1":
                print(f"Wprowadź {self.input_size} wartości wejściowych (oddzielone spacjami):")
                input_str = input().strip()
                inputs = [float(x) for x in input_str.split()]
                
                if len(inputs) != self.input_size:
                    print(f"Błąd: Oczekiwano {self.input_size} wartości, otrzymano {len(inputs)}")
                    return
                
                print(f"Wprowadź {self.output_size} wartości docelowych (oddzielone spacjami):")
                target_str = input().strip()
                targets = [float(x) for x in target_str.split()]
                
                if len(targets) != self.output_size:
                    print(f"Błąd: Oczekiwano {self.output_size} wartości, otrzymano {len(targets)}")
                    return
                    
                self.training_inputs.append(inputs)
                self.training_targets.append(targets)
                print("Dane dodane pomyślnie!")
                
            elif choice == "2":
                num_samples = int(input("Ile próbek wygenerować? "))
                for _ in range(num_samples):
                    inputs = np.random.randn(self.input_size).tolist()
                    targets = np.random.randn(self.output_size).tolist()
                    self.training_inputs.append(inputs)
                    self.training_targets.append(targets)
                print(f"Wygenerowano {num_samples} próbek treningowych")
                
            elif choice == "3":
                num_samples = int(input("Ile próbek z wzorcem wygenerować? "))
                print("Generowanie danych z wzorcami dla testowania DRM...")
                
                for i in range(num_samples):
                    # Twórz dane z różnymi wzorcami
                    if i % 4 == 0:  # Wysokie wariancje
                        inputs = (np.random.randn(self.input_size) * 3).tolist()
                    elif i % 4 == 1:  # Niskie wartości
                        inputs = (np.random.randn(self.input_size) - 2).tolist()
                    elif i % 4 == 2:  # Outliers
                        inputs = np.random.randn(self.input_size).tolist()
                        inputs[0] = 10.0  # Outlier
                    else:  # Normalne dane
                        inputs = np.random.randn(self.input_size).tolist()
                    
                    # Targets na podstawie wzorca
                    targets = [np.mean(inputs), np.std(inputs), np.max(inputs)][:self.output_size]
                    
                    self.training_inputs.append(inputs)
                    self.training_targets.append(targets)
                    
                print(f"Wygenerowano {num_samples} próbek z wzorcami")
            else:
                print("Nieprawidłowy wybór")
                return
                
        except ValueError as e:
            print(f"Błąd: Nieprawidłowy format danych - {e}")
        except Exception as e:
            print(f"Błąd: {e}")
    
    def train_network(self):
        """Trenowanie sieci z zaawansowanym DRM"""
        if not self.training_inputs or not self.training_targets:
            print("Brak danych treningowych! Najpierw dodaj dane.")
            return
            
        print("\n=== Zaawansowane trenowanie z DRM ===")
        
        try:
            epochs = int(input("Liczba epok (domyślnie 100): ") or "100")
            
            # Konwertuj na tensory
            inputs_tensor = torch.FloatTensor(self.training_inputs)
            targets_tensor = torch.FloatTensor(self.training_targets)
            
            print(f"Rozpoczynanie treningu na {len(self.training_inputs)} próbkach...")
            print(f"Tryb początkowy DRM: {self.drm.mode.value}")
            
            for epoch in range(epochs):
                # Forward pass
                outputs = self.model(inputs_tensor)
                loss = self.criterion(outputs, targets_tensor)
                
                # Oblicz dokładność predykcji
                accuracy = self.calculate_prediction_accuracy(outputs, targets_tensor)
                
                # Oblicz FRZ
                FRZ = self.rls.calculate_FRZ(loss.item(), accuracy)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Zastosuj zaawansowane reguły DRM dla każdego kontekstu
                drm_results = []
                for i, input_data in enumerate(self.training_inputs):
                    result = self.drm.apply_rules(input_data, FRZ)
                    drm_results.append(result)
                
                # Wykryj różnice używając RLS
                difference_detected = self.rls.detect_difference(loss.item())
                
                # Zapisz metryki
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'loss': loss.item(),
                    'accuracy': accuracy,
                    'FRZ': FRZ,
                    'system_mode': self.drm.mode.value,
                    'avg_rule_strength': self.drm.calculate_system_average_strength(),
                    'active_rules': len(self.drm.rules),
                    'difference_detected': difference_detected
                }
                self.training_history.append(epoch_metrics)
                
                # Wyświetl postęp
                if difference_detected or epoch % (epochs // 10) == 0 or epoch == epochs - 1:
                    mode_icon = "🔍" if self.drm.mode == SystemMode.EXPLORATION else "⚡"
                    diff_icon = "🔥" if difference_detected else ""
                    
                    print(f'Epoka {epoch+1:3d}/{epochs} | '
                          f'Loss: {loss.item():.6f} | '
                          f'FRZ: {FRZ:.3f} | '
                          f'Acc: {accuracy:.3f} | '
                          f'{mode_icon} {self.drm.mode.value} | '
                          f'Reguły: {len(self.drm.rules)} | '
                          f'S̄: {self.drm.calculate_system_average_strength():.3f} {diff_icon}')
                
                # Co 20 epok - pokaż szczegóły DRM
                if epoch > 0 and epoch % 20 == 0:
                    self._show_drm_status()
            
            print("\n✅ Trenowanie zakończone!")
            self._show_final_training_summary()
            
        except ValueError as e:
            print(f"Błąd: Nieprawidłowa liczba epok - {e}")
        except Exception as e:
            print(f"Błąd podczas trenowania: {e}")
    
    def _show_drm_status(self):
        """Pokaż aktualny status DRM"""
        print(f"\n--- Status DRM (T={self.drm.T}) ---")
        stats = self.drm.get_detailed_statistics()
        
        print(f"Tryb systemu: {stats['system_mode']} | "
              f"Średnia siła: {stats['average_strength']:.3f} | "
              f"Reguły: {stats['total_rules']}")
        
        # Pokaż top 3 najsilniejsze reguły
        rule_strengths = [(rule_id, data['strength']) 
                         for rule_id, data in stats['rules'].items()]
        rule_strengths.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 3 najsilniejsze reguły:")
        for i, (rule_id, strength) in enumerate(rule_strengths[:3]):
            rule_data = stats['rules'][rule_id]
            print(f"  {i+1}. {rule_id}: S={strength:.3f} "
                  f"(W={rule_data['effectiveness_weight']:.2f}, "
                  f"C={rule_data['context_range']:.2f}, "
                  f"R={rule_data['resonance']:.2f})")
    
    def _show_final_training_summary(self):
        """Pokaż podsumowanie trenowania"""
        if not self.training_history:
            return
            
        print("\n=== Podsumowanie trenowania ===")
        
        # Podstawowe statystyki
        final_metrics = self.training_history[-1]
        initial_loss = self.training_history[0]['loss']
        final_loss = final_metrics['loss']
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"📊 Poprawa loss: {improvement:.1f}% "
              f"({initial_loss:.6f} → {final_loss:.6f})")
        print(f"🎯 Końcowy FRZ: {final_metrics['FRZ']:.3f}")
        print(f"🔧 Tryb końcowy: {final_metrics['system_mode']}")
        print(f"📋 Aktywne reguły: {final_metrics['active_rules']}")
        
        # Statystyki RLS
        rls_stats = self.rls.get_statistics()
        print(f"🔍 Wykryte różnice: {rls_stats['differences_detected']}")
        print(f"📈 Trend FRZ: {rls_stats['FRZ_trend']}")
        
        # Statystyki DRM
        drm_stats = self.drm.get_detailed_statistics()
        print(f"⚡ Średnia siła systemu: {drm_stats['average_strength']:.3f}")
        
        # Analiza ewolucji reguł
        mode_changes = 0
        exploration_epochs = 0
        for i in range(1, len(self.training_history)):
            if (self.training_history[i]['system_mode'] != 
                self.training_history[i-1]['system_mode']):
                mode_changes += 1
            if self.training_history[i]['system_mode'] == 'exploration':
                exploration_epochs += 1
        
        print(f"🔄 Zmiany trybu: {mode_changes}")
        print(f"🔍 Epoki eksploracji: {exploration_epochs}/{len(self.training_history)} "
              f"({exploration_epochs/len(self.training_history)*100:.1f}%)")
    
    def save_model(self):
        """Zapisz model z pełnymi danymi DRM"""
        print("\n=== Zapisywanie zaawansowanego modelu ===")
        
        try:
            filename = input("Nazwa pliku (bez rozszerzenia, domyślnie 'advanced_model'): ").strip() or "advanced_model"
            
            # Przygotuj dane DRM do zapisu
            drm_data = {
                'rules_metadata': {},
                'system_time': self.drm.T,
                'mode': self.drm.mode.value,
                'adaptation_threshold': self.drm.adaptation_threshold,
                'exploration_threshold': self.drm.exploration_threshold,
                'memory': self.drm.memory,
                'rule_combinations': self.drm.rule_combinations
            }
            
            # Zapisz metadane reguł (bez funkcji callable)
            for rule_id, rule in self.drm.rules.items():
                meta = rule['metadata']
                drm_data['rules_metadata'][rule_id] = {
                    'W': meta.W,
                    'C': meta.C,
                    'U': meta.U,
                    'R': meta.R,
                    'creation_time': meta.creation_time,
                    'last_activation': meta.last_activation,
                    'success_count': meta.success_count,
                    'total_activations': meta.total_activations
                }
            
            # Kompletne dane modelu
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'training_inputs': self.training_inputs,
                'training_targets': self.training_targets,
                'training_history': self.training_history,
                'drm_data': drm_data,
                'rls_data': {
                    'threshold': self.rls.threshold,
                    'window_size': self.rls.window_size,
                    'history': self.rls.history,
                    'differences': self.rls.differences,
                    'FRZ_history': self.rls.FRZ_history
                }
            }
            
            torch.save(model_data, f"{filename}.pth")
            print(f"✅ Zaawansowany model zapisany jako {filename}.pth")
            print(f"📊 Zapisano {len(self.training_history)} epok historii trenowania")
            print(f"🔧 Zapisano {len(drm_data['rules_metadata'])} reguł DRM")
            
        except Exception as e:
            print(f"❌ Błąd podczas zapisywania: {e}")
    
    def load_model(self):
        """Wczytaj model z danymi DRM"""
        print("\n=== Wczytywanie zaawansowanego modelu ===")
        
        try:
            filename = input("Nazwa pliku (bez rozszerzenia): ").strip()
            
            if not os.path.exists(f"{filename}.pth"):
                print(f"❌ Plik {filename}.pth nie istnieje!")
                return
            
            model_data = torch.load(f"{filename}.pth")
            
            # Odtwórz architekturę sieci
            self.input_size = model_data['input_size']
            self.hidden_size = model_data['hidden_size']
            self.output_size = model_data['output_size']
            
            self.model = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Wczytaj stany
            self.model.load_state_dict(model_data['model_state_dict'])
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
            
            # Wczytaj dane treningowe
            self.training_inputs = model_data.get('training_inputs', [])
            self.training_targets = model_data.get('training_targets', [])
            self.training_history = model_data.get('training_history', [])
            
            # Odtwórz DRM
            if 'drm_data' in model_data:
                drm_data = model_data['drm_data']
                self.drm.T = drm_data['system_time']
                self.drm.mode = SystemMode(drm_data['mode'])
                self.drm.adaptation_threshold = drm_data['adaptation_threshold']
                self.drm.exploration_threshold = drm_data['exploration_threshold']
                self.drm.memory = drm_data['memory']
                self.drm.rule_combinations = drm_data.get('rule_combinations', [])
                
                # Odtwórz metadane reguł (reguły będą ponownie zainicjalizowane)
                self._initialize_advanced_rules()  # Odtwórz funkcje reguł
                
                # Przywróć metadane
                for rule_id, meta_data in drm_data['rules_metadata'].items():
                    if rule_id in self.drm.rules:
                        meta = self.drm.rules[rule_id]['metadata']
                        meta.W = meta_data['W']
                        meta.C = meta_data['C']
                        meta.U = meta_data['U']
                        meta.R = meta_data['R']
                        meta.creation_time = meta_data['creation_time']
                        meta.last_activation = meta_data['last_activation']
                        meta.success_count = meta_data['success_count']
                        meta.total_activations = meta_data['total_activations']
            
            # Odtwórz RLS
            if 'rls_data' in model_data:
                rls_data = model_data['rls_data']
                self.rls.threshold = rls_data['threshold']
                self.rls.window_size = rls_data['window_size']
                self.rls.history = rls_data['history']
                self.rls.differences = rls_data['differences']
                self.rls.FRZ_history = rls_data['FRZ_history']
            
            print(f"✅ Model wczytany z {filename}.pth")
            print(f"🏗️ Architektura: {self.input_size} → {self.hidden_size} → {self.output_size}")
            print(f"📊 Dane treningowe: {len(self.training_inputs)} próbek")
            print(f"📈 Historia: {len(self.training_history)} epok")
            print(f"🔧 DRM: {len(self.drm.rules)} reguł, T={self.drm.T}, tryb={self.drm.mode.value}")
            print(f"🎯 RLS: FRZ={self.rls.get_current_FRZ():.3f}")
            
        except Exception as e:
            print(f"❌ Błąd podczas wczytywania: {e}")
    
    def test_model(self):
        """Testuj model z analizą DRM"""
        print("\n=== Testowanie modelu z analizą DRM ===")
        
        try:
            print(f"Wprowadź {self.input_size} wartości testowych (oddzielone spacjami):")
            input_str = input().strip()
            test_input = [float(x) for x in input_str.split()]
            
            if len(test_input) != self.input_size:
                print(f"❌ Błąd: Oczekiwano {self.input_size} wartości, otrzymano {len(test_input)}")
                return
            
            # Predykcja
            with torch.no_grad():
                input_tensor = torch.FloatTensor([test_input])
                output = self.model(input_tensor)
                prediction = output[0].tolist()
            
            print(f"🎯 Predykcja: {[f'{x:.4f}' for x in prediction]}")
            
            # Analiza DRM dla danych testowych
            print("\n--- Analiza DRM ---")
            current_FRZ = self.rls.get_current_FRZ()
            drm_result = self.drm.apply_rules(test_input, current_FRZ)
            
            print(f"🎯 Aktualny FRZ: {current_FRZ:.3f}")
            print(f"🔧 Tryb systemu: {drm_result['system_mode']}")
            print(f"⚡ Średnia siła systemu: {drm_result['average_strength']:.3f}")
            print(f"📋 Zastosowane reguły: {len(drm_result['applied_rules'])}")
            
            if drm_result['applied_rules']:
                print("Aktywne reguły:")
                for rule_id in drm_result['applied_rules']:
                    strength = drm_result['rule_strengths'][rule_id]
                    print(f"  • {rule_id}: siła = {strength:.3f}")
            else:
                print("  Brak aktywnych reguł dla tego kontekstu")
            
            # Analiza kontekstu
            print(f"\n--- Analiza kontekstu ---")
            context_stats = {
                'mean': np.mean(test_input),
                'std': np.std(test_input),
                'var': np.var(test_input),
                'min': np.min(test_input),
                'max': np.max(test_input),
                'range': np.max(test_input) - np.min(test_input)
            }
            
            for stat_name, value in context_stats.items():
                print(f"  {stat_name}: {value:.4f}")
            
            # Porównanie z danymi treningowymi
            if self.training_inputs:
                print(f"\n--- Porównanie z danymi treningowymi ---")
                training_means = [np.mean(inp) for inp in self.training_inputs]
                training_stds = [np.std(inp) for inp in self.training_inputs]
                
                mean_similarity = 1.0 / (1.0 + abs(context_stats['mean'] - np.mean(training_means)))
                std_similarity = 1.0 / (1.0 + abs(context_stats['std'] - np.mean(training_stds)))
                
                print(f"  Podobieństwo średniej: {mean_similarity:.3f}")
                print(f"  Podobieństwo odchylenia: {std_similarity:.3f}")
                
                if mean_similarity < 0.5 or std_similarity < 0.5:
                    print("  ⚠️ Dane testowe znacznie różnią się od treningowych!")
            
        except ValueError as e:
            print(f"❌ Błąd: Nieprawidłowy format danych - {e}")
        except Exception as e:
            print(f"❌ Błąd podczas testowania: {e}")
    
    def show_advanced_statistics(self):
        """Pokaż zaawansowane statystyki systemu"""
        print("\n" + "="*60)
        print("    ZAAWANSOWANE STATYSTYKI SYSTEMU LOOPDRM")
        print("="*60)
        
        # Statystyki sieci neuronowej
        print(f"\n🧠 SIEĆ NEURONOWA")
        print(f"Architektura: {self.input_size} → {self.hidden_size} → {self.output_size}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parametry: {total_params} (treningowe: {trainable_params})")
        
        if self.training_history:
            print(f"Epoki trenowania: {len(self.training_history)}")
            final_loss = self.training_history[-1]['loss']
            initial_loss = self.training_history[0]['loss']
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"Poprawa loss: {improvement:.2f}%")
        
        # Zaawansowane statystyki DRM
        print(f"\n🔧 DYNAMICZNA MATRYCA REGUŁ (DRM)")
        drm_stats = self.drm.get_detailed_statistics()
        
        print(f"Czas systemowy (T): {drm_stats['system_time']}")
        print(f"Tryb systemu: {drm_stats['system_mode']}")
        print(f"Średnia siła systemu (S̄): {drm_stats['average_strength']:.4f}")
        print(f"Próg adaptacji: {drm_stats['adaptation_threshold']:.3f}")
        print(f"Próg eksploracji: {drm_stats['exploration_threshold']:.3f}")
        print(f"Rozmiar pamięci: {drm_stats['memory_size']}")
        
        print(f"\n📊 SZCZEGÓŁY REGUŁ ({drm_stats['total_rules']} aktywnych):")
        print("-" * 80)
        print(f"{'ID Reguły':<20} {'Siła(Si)':<10} {'W':<8} {'C':<8} {'U':<8} {'R':<8} {'Sukces%':<10} {'Wiek':<6}")
        print("-" * 80)
        
        # Sortuj reguły według siły
        rules_by_strength = sorted(drm_stats['rules'].items(), 
                                 key=lambda x: x[1]['strength'], reverse=True)
        
        for rule_id, rule_data in rules_by_strength:
            print(f"{rule_id:<20} "
                  f"{rule_data['strength']:<10.3f} "
                  f"{rule_data['effectiveness_weight']:<8.2f} "
                  f"{rule_data['context_range']:<8.2f} "
                  f"{rule_data['usage']:<8.1f} "
                  f"{rule_data['resonance']:<8.3f} "
                  f"{rule_data['success_rate']*100:<10.1f} "
                  f"{rule_data['age']:<6}")
        
        # Statystyki RLS
        print(f"\n🎯 SYSTEM UCZENIA REGUŁ (RLS)")
        rls_stats = self.rls.get_statistics()
        
        print(f"Próg detekcji: {rls_stats['threshold']}")
        print(f"Rozmiar okna: {self.rls.window_size}")
        print(f"Historia: {rls_stats['history_length']} punktów")
        print(f"Wykryte różnice: {rls_stats['differences_detected']}")
        print(f"Aktualny FRZ: {rls_stats['current_FRZ']:.4f}")
        print(f"Średni FRZ: {rls_stats['average_FRZ']:.4f}")
        print(f"Trend FRZ: {rls_stats['FRZ_trend']}")
        
        # Analiza historii trenowania
        if self.training_history:
            print(f"\n📈 ANALIZA HISTORII TRENOWANIA")
            
            # Statystyki trybów
            exploration_count = sum(1 for h in self.training_history 
                                  if h['system_mode'] == 'exploration')
            exploitation_count = len(self.training_history) - exploration_count
            
            print(f"Epoki eksploracji: {exploration_count} ({exploration_count/len(self.training_history)*100:.1f}%)")
            print(f"Epoki eksploatacji: {exploitation_count} ({exploitation_count/len(self.training_history)*100:.1f}%)")
            
            # Zmiany trybu
            mode_changes = 0
            for i in range(1, len(self.training_history)):
                if (self.training_history[i]['system_mode'] != 
                    self.training_history[i-1]['system_mode']):
                    mode_changes += 1
            
            print(f"Zmiany trybu: {mode_changes}")
            
            # Statystyki FRZ
            frz_values = [h['FRZ'] for h in self.training_history]
            print(f"FRZ - min: {min(frz_values):.3f}, max: {max(frz_values):.3f}, "
                  f"średni: {np.mean(frz_values):.3f}")
            
            # Ewolucja liczby reguł
            rule_counts = [h['active_rules'] for h in self.training_history]
            print(f"Reguły - min: {min(rule_counts)}, max: {max(rule_counts)}, "
                  f"końcowa: {rule_counts[-1]}")
            
            # Wykryte różnice
            differences = sum(1 for h in self.training_history if h['difference_detected'])
            print(f"Wykryte różnice: {differences} ({differences/len(self.training_history)*100:.1f}%)")
        
        # Analiza wzorców matematycznych
        print(f"\n🔬 ANALIZA WZORCÓW MATEMATYCZNYCH")
        
        if drm_stats['rules']:
            # Korelacje między parametrami reguł
            W_values = [data['effectiveness_weight'] for data in drm_stats['rules'].values()]
            C_values = [data['context_range'] for data in drm_stats['rules'].values()]
            R_values = [data['resonance'] for data in drm_stats['rules'].values()]
            S_values = [data['strength'] for data in drm_stats['rules'].values()]
            
            print(f"Rozkład wag skuteczności (W): min={min(W_values):.2f}, "
                  f"max={max(W_values):.2f}, średnia={np.mean(W_values):.2f}")
            print(f"Rozkład zasięgu kontekstowego (C): min={min(C_values):.2f}, "
                  f"max={max(C_values):.2f}, średnia={np.mean(C_values):.2f}")
            print(f"Rozkład rezonansu (R): min={min(R_values):.3f}, "
                  f"max={max(R_values):.3f}, średnia={np.mean(R_values):.3f}")
            print(f"Rozkład siły (S): min={min(S_values):.3f}, "
                  f"max={max(S_values):.3f}, średnia={np.mean(S_values):.3f}")
            
            # Sprawdź wzór Si = Wi · log(Ci + 1) · (1 + Ui/T) · Ri
            print(f"\n🧮 WERYFIKACJA WZORU MATEMATYCZNEGO:")
            print("Si = Wi · log(Ci + 1) · (1 + Ui/T) · Ri")
            
            for rule_id, rule_data in list(drm_stats['rules'].items())[:3]:  # Pokaż 3 przykłady
                W = rule_data['effectiveness_weight']
                C = rule_data['context_range']
                U = rule_data['usage']
                R = rule_data['resonance']
                T = max(drm_stats['system_time'], 1)
                
                calculated_S = W * math.log(C + 1) * (1 + U/T) * R
                actual_S = rule_data['strength']
                
                print(f"  {rule_id}: obliczone={calculated_S:.4f}, "
                      f"rzeczywiste={actual_S:.4f}, różnica={abs(calculated_S-actual_S):.6f}")
        
        print("\n" + "="*60)
    
    def run_advanced_menu(self):
        """Zaawansowane menu główne"""
        while True:
            print("\n" + "="*60)
            print("    🚀 ZAAWANSOWANY LOOPDRM - SYSTEM DRM Z ZAMKNIĘTĄ PĘTLĄ")
            print("="*60)
            print("1. 📝 Wprowadzenie danych treningowych")
            print("2. 🎯 Trenowanie sieci z DRM")
            print("3. 💾 Zapisanie modelu")
            print("4. 📂 Wczytanie modelu")
            print("5. 🧪 Testowanie modelu")
            print("6. 📊 Zaawansowane statystyki")
            print("7. 🔧 Konfiguracja DRM")
            print("8. 📈 Analiza wydajności")
            print("9. 🔍 Eksport danych")
            print("0. 🚪 Zakończenie")
            print("-" * 60)
            
            try:
                choice = input("Wybierz opcję (0-9): ").strip()
                
                if choice == "1":
                    self.add_training_data()
                elif choice == "2":
                    self.train_network()
                elif choice == "3":
                    self.save_model()
                elif choice == "4":
                    self.load_model()
                elif choice == "5":
                    self.test_model()
                elif choice == "6":
                    self.show_advanced_statistics()
                elif choice == "7":
                    self.configure_drm()
                elif choice == "8":
                    self.analyze_performance()
                elif choice == "9":
                    self.export_data()
                elif choice == "0":
                    print("🎉 Dziękujemy za korzystanie z Zaawansowanego LoopDRM!")
                    print("📊 System wykonał {} iteracji DRM".format(self.drm.T))
                    if self.training_history:
                        print("🏆 Najlepszy FRZ: {:.4f}".format(
                            max(h['FRZ'] for h in self.training_history)))
                    break
                else:
                    print("❌ Nieprawidłowy wybór. Wybierz opcję 0-9.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Program przerwany przez użytkownika.")
                break
            except Exception as e:
                print(f"❌ Wystąpił nieoczekiwany błąd: {e}")
    
    def configure_drm(self):
        """Konfiguracja parametrów DRM"""
        print("\n=== Konfiguracja DRM ===")
        
        try:
            print(f"Aktualne ustawienia:")
            print(f"  Próg adaptacji: {self.drm.adaptation_threshold}")
            print(f"  Próg eksploracji: {self.drm.exploration_threshold}")
            print(f"  Tryb systemu: {self.drm.mode.value}")
            
            print("\nCo chcesz zmienić?")
            print("1. Próg adaptacji")
            print("2. Próg eksploracji")
            print("3. Wymuszenie trybu systemu")
            print("4. Parametry RLS")
            print("5. Reset systemu DRM")
            print("0. Powrót")
            
            config_choice = input("Wybierz opcję: ").strip()
            
            if config_choice == "1":
                new_threshold = float(input(f"Nowy próg adaptacji (aktualny: {self.drm.adaptation_threshold}): "))
                if 0.0 <= new_threshold <= 1.0:
                    self.drm.adaptation_threshold = new_threshold
                    print(f"✅ Próg adaptacji zmieniony na {new_threshold}")
                else:
                    print("❌ Próg musi być w zakresie 0.0-1.0")
                    
            elif config_choice == "2":
                new_threshold = float(input(f"Nowy próg eksploracji (aktualny: {self.drm.exploration_threshold}): "))
                if 0.0 <= new_threshold <= 2.0:
                    self.drm.exploration_threshold = new_threshold
                    print(f"✅ Próg eksploracji zmieniony na {new_threshold}")
                else:
                    print("❌ Próg musi być w zakresie 0.0-2.0")
                    
            elif config_choice == "3":
                print("1. Eksploracja")
                print("2. Eksploatacja")
                mode_choice = input("Wybierz tryb: ").strip()
                
                if mode_choice == "1":
                    self.drm.mode = SystemMode.EXPLORATION
                    print("✅ Wymuszono tryb eksploracji")
                elif mode_choice == "2":
                    self.drm.mode = SystemMode.EXPLOITATION
                    print("✅ Wymuszono tryb eksploatacji")
                else:
                    print("❌ Nieprawidłowy wybór")
                    
            elif config_choice == "4":
                print(f"Aktualne parametry RLS:")
                print(f"  Próg: {self.rls.threshold}")
                print(f"  Rozmiar okna: {self.rls.window_size}")
                
                new_rls_threshold = float(input(f"Nowy próg RLS (aktualny: {self.rls.threshold}): "))
                new_window_size = int(input(f"Nowy rozmiar okna (aktualny: {self.rls.window_size}): "))
                
                if new_rls_threshold > 0 and new_window_size > 0:
                    self.rls.threshold = new_rls_threshold
                    self.rls.window_size = new_window_size
                    print("✅ Parametry RLS zaktualizowane")
                else:
                    print("❌ Parametry muszą być dodatnie")
                    
            elif config_choice == "5":
                confirm = input("⚠️ Czy na pewno chcesz zresetować system DRM? (tak/nie): ").strip().lower()
                if confirm in ['tak', 'yes', 'y']:
                    self.drm = AdvancedDRM(self.drm.adaptation_threshold, self.drm.exploration_threshold)
                    self._initialize_advanced_rules()
                    print("✅ System DRM został zresetowany")
                else:
                    print("❌ Reset anulowany")
                    
            elif config_choice == "0":
                return
            else:
                print("❌ Nieprawidłowy wybór")
                
        except ValueError as e:
            print(f"❌ Błąd: Nieprawidłowa wartość - {e}")
        except Exception as e:
            print(f"❌ Błąd konfiguracji: {e}")
    
    def analyze_performance(self):
        """Analiza wydajności systemu"""
        print("\n=== Analiza wydajności systemu ===")
        
        if not self.training_history:
            print("❌ Brak danych historycznych do analizy")
            return
        
        try:
            # Analiza konwergencji
            print("📈 ANALIZA KONWERGENCJI")
            losses = [h['loss'] for h in self.training_history]
            frz_values = [h['FRZ'] for h in self.training_history]
            
            # Oblicz trendy
            if len(losses) >= 10:
                recent_losses = losses[-10:]
                early_losses = losses[:10]
                
                recent_avg = np.mean(recent_losses)
                early_avg = np.mean(early_losses)
                improvement = ((early_avg - recent_avg) / early_avg) * 100
                
                print(f"  Poprawa loss (ostatnie 10 vs pierwsze 10): {improvement:.2f}%")
                
                # Stabilność
                recent_std = np.std(recent_losses)
                print(f"  Stabilność (odchylenie ostatnich 10): {recent_std:.6f}")
                
                if recent_std < 0.001:
                    print("  ✅ System osiągnął wysoką stabilność")
                elif recent_std < 0.01:
                    print("  ⚠️ System jest umiarkowanie stabilny")
                else:
                    print("  ❌ System jest niestabilny")
            
            # Analiza efektywności DRM
            print(f"\n🔧 ANALIZA EFEKTYWNOŚCI DRM")
            
            rule_strength_history = [h['avg_rule_strength'] for h in self.training_history]
            rule_count_history = [h['active_rules'] for h in self.training_history]
            
            print(f"  Średnia siła reguł: {np.mean(rule_strength_history):.4f}")
            print(f"  Stabilność siły reguł: {np.std(rule_strength_history):.4f}")
            print(f"  Średnia liczba reguł: {np.mean(rule_count_history):.1f}")
            print(f"  Zmienność liczby reguł: {np.std(rule_count_history):.1f}")
            
            # Analiza trybów
            exploration_epochs = sum(1 for h in self.training_history if h['system_mode'] == 'exploration')
            exploitation_epochs = len(self.training_history) - exploration_epochs
            
            print(f"\n🔍 ANALIZA TRYBÓW SYSTEMU")
            print(f"  Eksploracja: {exploration_epochs} epok ({exploration_epochs/len(self.training_history)*100:.1f}%)")
            print(f"  Eksploatacja: {exploitation_epochs} epok ({exploitation_epochs/len(self.training_history)*100:.1f}%)")
            
            # Efektywność trybów
            exploration_frz = [h['FRZ'] for h in self.training_history if h['system_mode'] == 'exploration']
            exploitation_frz = [h['FRZ'] for h in self.training_history if h['system_mode'] == 'exploitation']
            
            if exploration_frz and exploitation_frz:
                print(f"  Średni FRZ w eksploracji: {np.mean(exploration_frz):.4f}")
                print(f"  Średni FRZ w eksploatacji: {np.mean(exploitation_frz):.4f}")
                
                if np.mean(exploitation_frz) > np.mean(exploration_frz):
                    print("  ✅ Eksploatacja jest bardziej efektywna")
                else:
                    print("  ⚠️ Eksploracja daje lepsze wyniki")
            
            # Analiza różnic wykrytych przez RLS
            differences_detected = sum(1 for h in self.training_history if h['difference_detected'])
            print(f"\n🎯 ANALIZA RLS")
            print(f"  Wykryte różnice: {differences_detected} ({differences_detected/len(self.training_history)*100:.1f}%)")
            
            if differences_detected > 0:
                # Znajdź epoki z różnicami
                diff_epochs = [i for i, h in enumerate(self.training_history) if h['difference_detected']]
                
                # Sprawdź czy różnice korelują z poprawą
                improvements_after_diff = 0
                for epoch in diff_epochs:
                    if epoch < len(self.training_history) - 5:  # Sprawdź 5 epok później
                        before_frz = self.training_history[epoch]['FRZ']
                        after_frz = np.mean([self.training_history[i]['FRZ'] 
                                           for i in range(epoch+1, min(epoch+6, len(self.training_history)))])
                        if after_frz > before_frz:
                            improvements_after_diff += 1
                
                improvement_rate = improvements_after_diff / len(diff_epochs) * 100
                print(f"  Poprawa po wykryciu różnic: {improvement_rate:.1f}%")
                
                if improvement_rate > 60:
                    print("  ✅ RLS skutecznie wykrywa momenty poprawy")
                else:
                    print("  ⚠️ RLS może wymagać dostrojenia")
            
            # Rekomendacje
            print(f"\n💡 REKOMENDACJE")
            
            if improvement < 10:
                print("  • Rozważ zwiększenie learning rate lub liczby epok")
            
            if recent_std > 0.01:
                print("  • System jest niestabilny - rozważ zmniejszenie learning rate")
            
            if np.mean(rule_strength_history) < 0.5:
                print("  • Niska siła reguł - rozważ zmniejszenie progu adaptacji")
            
            if exploration_epochs / len(self.training_history) > 0.7:
                print("  • Zbyt dużo eksploracji - rozważ zwiększenie progu eksploracji")
            elif exploration_epochs / len(self.training_history) < 0.1:
                print("  • Zbyt mało eksploracji - rozważ zmniejszenie progu eksploracji")
            
            if differences_detected / len(self.training_history) > 0.5:
                print("  • RLS wykrywa zbyt wiele różnic - rozważ zwiększenie progu")
            elif differences_detected / len(self.training_history) < 0.05:
                print("  • RLS wykrywa zbyt mało różnic - rozważ zmniejszenie progu")
                
        except Exception as e:
            print(f"❌ Błąd podczas analizy: {e}")
    
    def export_data(self):
        """Eksport danych do analizy"""
        print("\n=== Eksport danych ===")
        
        try:
            print("1. Eksport historii trenowania (JSON)")
            print("2. Eksport statystyk DRM (JSON)")
            print("3. Eksport danych do CSV")
            print("4. Eksport pełnego raportu (TXT)")
            print("0. Powrót")
            
            export_choice = input("Wybierz opcję: ").strip()
            
            if export_choice == "1":
                filename = input("Nazwa pliku (bez rozszerzenia): ").strip() or "training_history"
                
                with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                    json.dump(self.training_history, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Historia trenowania zapisana do {filename}.json")
                
            elif export_choice == "2":
                filename = input("Nazwa pliku (bez rozszerzenia): ").strip() or "drm_statistics"
                
                drm_stats = self.drm.get_detailed_statistics()
                
                with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                    json.dump(drm_stats, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Statystyki DRM zapisane do {filename}.json")
                
            elif export_choice == "3":
                filename = input("Nazwa pliku (bez rozszerzenia): ").strip() or "training_data"
                
                if not self.training_history:
                    print("❌ Brak danych do eksportu")
                    return
                
                import csv
                
                with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Nagłówki
                    headers = ['epoch', 'loss', 'accuracy', 'FRZ', 'system_mode', 
                              'avg_rule_strength', 'active_rules', 'difference_detected']
                    writer.writerow(headers)
                    
                    # Dane
                    for h in self.training_history:
                        row = [h.get(header, '') for header in headers]
                        writer.writerow(row)
                
                print(f"✅ Dane treningowe zapisane do {filename}.csv")
                
            elif export_choice == "4":
                filename = input("Nazwa pliku (bez rozszerzenia): ").strip() or "full_report"
                
                with open(f"{filename}.txt", 'w', encoding='utf-8') as f:
                    f.write("PEŁNY RAPORT SYSTEMU LOOPDRM\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Podstawowe informacje
                    f.write(f"Data generowania: {np.datetime64('now')}\n")
                    f.write(f"Architektura sieci: {self.input_size} → {self.hidden_size} → {self.output_size}\n")
                    f.write(f"Liczba próbek treningowych: {len(self.training_inputs)}\n")
                    f.write(f"Liczba epok: {len(self.training_history)}\n\n")
                    
                    # Statystyki DRM
                    drm_stats = self.drm.get_detailed_statistics()
                    f.write("STATYSTYKI DRM:\n")
                    f.write(f"Czas systemowy: {drm_stats['system_time']}\n")
                    f.write(f"Tryb: {drm_stats['system_mode']}\n")
                    f.write(f"Średnia siła: {drm_stats['average_strength']:.4f}\n")
                    f.write(f"Liczba reguł: {drm_stats['total_rules']}\n")
                    f.write(f"Próg adaptacji: {drm_stats['adaptation_threshold']}\n")
                    f.write(f"Próg eksploracji: {drm_stats['exploration_threshold']}\n\n")
                    
                    # Szczegóły reguł
                    f.write("SZCZEGÓŁY REGUŁ:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"{'ID':<20} {'Siła':<10} {'W':<8} {'C':<8} {'U':<8} {'R':<8} {'Sukces%':<10} {'Wiek':<6}\n")
                    f.write("-" * 80 + "\n")
                    
                    for rule_id, rule_data in drm_stats['rules'].items():
                        f.write(f"{rule_id:<20} "
                               f"{rule_data['strength']:<10.3f} "
                               f"{rule_data['effectiveness_weight']:<8.2f} "
                               f"{rule_data['context_range']:<8.2f} "
                               f"{rule_data['usage']:<8.1f} "
                               f"{rule_data['resonance']:<8.3f} "
                               f"{rule_data['success_rate']*100:<10.1f} "
                               f"{rule_data['age']:<6}\n")
                    
                    # Statystyki RLS
                    rls_stats = self.rls.get_statistics()
                    f.write(f"\nSTATYSTYKI RLS:\n")
                    f.write(f"Próg: {rls_stats['threshold']}\n")
                    f.write(f"Rozmiar okna: {self.rls.window_size}\n")
                    f.write(f"Wykryte różnice: {rls_stats['differences_detected']}\n")
                    f.write(f"Aktualny FRZ: {rls_stats['current_FRZ']:.4f}\n")
                    f.write(f"Średni FRZ: {rls_stats['average_FRZ']:.4f}\n")
                    f.write(f"Trend FRZ: {rls_stats['FRZ_trend']}\n\n")
                    
                    # Historia trenowania (ostatnie 20 epok)
                    if self.training_history:
                        f.write("OSTATNIE 20 EPOK TRENOWANIA:\n")
                        f.write("-" * 60 + "\n")
                        f.write(f"{'Epoka':<8} {'Loss':<12} {'FRZ':<8} {'Tryb':<12} {'Reguły':<8}\n")
                        f.write("-" * 60 + "\n")
                        
                        for h in self.training_history[-20:]:
                            f.write(f"{h['epoch']:<8} "
                                   f"{h['loss']:<12.6f} "
                                   f"{h['FRZ']:<8.3f} "
                                   f"{h['system_mode']:<12} "
                                   f"{h['active_rules']:<8}\n")
                
                print(f"✅ Pełny raport zapisany do {filename}.txt")
                
            elif export_choice == "0":
                return
            else:
                print("❌ Nieprawidłowy wybór")
                
        except Exception as e:
            print(f"❌ Błąd podczas eksportu: {e}")

# Funkcja główna do uruchomienia systemu
def main():
    """Funkcja główna uruchamiająca zaawansowany system LoopDRM"""
    print("🚀 Inicjalizacja Zaawansowanego Systemu LoopDRM...")
    
    try:
        # Zapytaj o konfigurację sieci
        print("\n=== Konfiguracja sieci neuronowej ===")
        input_size = int(input("Rozmiar warstwy wejściowej (domyślnie 10): ") or "10")
        hidden_size = int(input("Rozmiar warstwy ukrytej (domyślnie 20): ") or "20")
        output_size = int(input("Rozmiar warstwy wyjściowej (domyślnie 3): ") or "3")
        
        # Inicjalizuj system
        system = AdvancedLoopDRMSystem(input_size, hidden_size, output_size)
        
        print(f"\n✅ System zainicjalizowany!")
        print(f"🧠 Architektura sieci: {input_size} → {hidden_size} → {output_size}")
        print(f"🔧 DRM: {len(system.drm.rules)} reguł początkowych")
        print(f"🎯 RLS: próg = {system.rls.threshold}, okno = {system.rls.window_size}")
        
        # Uruchom menu główne
        system.run_advanced_menu()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Program przerwany przez użytkownika.")
    except ValueError as e:
        print(f"❌ Błąd konfiguracji: {e}")
        print("Używam domyślnych wartości...")
        system = AdvancedLoopDRMSystem()
        system.run_advanced_menu()
    except Exception as e:
        print(f"❌ Krytyczny błąd systemu: {e}")
        print("Sprawdź instalację wymaganych bibliotek:")
        print("pip install torch numpy")

# Dodatkowe funkcje pomocnicze
def create_demo_system():
    """Utwórz system demonstracyjny z przykładowymi danymi"""
    print("🎭 Tworzenie systemu demonstracyjnego...")
    
    system = AdvancedLoopDRMSystem(input_size=5, hidden_size=10, output_size=2)
    
    # Dodaj przykładowe dane
    np.random.seed(42)  # Dla powtarzalności
    for i in range(50):
        # Różne wzorce danych
        if i % 3 == 0:  # Wzorzec liniowy
            inputs = np.linspace(-2, 2, 5).tolist()
            targets = [np.sum(inputs), np.mean(inputs)]
        elif i % 3 == 1:  # Wzorzec kwadratowy
            inputs = (np.random.randn(5) ** 2).tolist()
            targets = [np.max(inputs), np.std(inputs)]
        else:  # Wzorzec losowy
            inputs = np.random.randn(5).tolist()
            targets = [np.median(inputs), np.var(inputs)]
        
        system.training_inputs.append(inputs)
        system.training_targets.append(targets)
    
    print(f"✅ System demonstracyjny utworzony z {len(system.training_inputs)} próbkami")
    return system

def run_benchmark():
    """Uruchom test wydajności systemu"""
    print("⏱️ Uruchamianie testu wydajności...")
    
    import time
    
    system = AdvancedLoopDRMSystem(input_size=8, hidden_size=16, output_size=4)
    
    # Generuj dane testowe
    start_time = time.time()
    
    for i in range(100):
        inputs = np.random.randn(8).tolist()
        targets = np.random.randn(4).tolist()
        system.training_inputs.append(inputs)
        system.training_targets.append(targets)
    
    data_gen_time = time.time() - start_time
    
    # Test trenowania
    start_time = time.time()
    
    inputs_tensor = torch.FloatTensor(system.training_inputs)
    targets_tensor = torch.FloatTensor(system.training_targets)
    
    for epoch in range(50):
        outputs = system.model(inputs_tensor)
        loss = system.criterion(outputs, targets_tensor)
        
        system.optimizer.zero_grad()
        loss.backward()
        system.optimizer.step()
        
        # Symuluj DRM
        FRZ = system.rls.calculate_FRZ(loss.item())
        for input_data in system.training_inputs[:5]:  # Tylko pierwsze 5 dla szybkości
            system.drm.apply_rules(input_data, FRZ)
    
    training_time = time.time() - start_time
    
    print(f"📊 Wyniki benchmarku:")
    print(f"  Generowanie danych: {data_gen_time:.4f}s")
    print(f"  Trenowanie (50 epok): {training_time:.4f}s")
    print(f"  Średni czas na epokę: {training_time/50:.4f}s")
    print(f"  Reguły DRM: {len(system.drm.rules)}")
    print(f"  Czas systemowy DRM: {system.drm.T}")

# Uruchomienie programu
if __name__ == "__main__":
    print("🌟 Witaj w Zaawansowanym Systemie LoopDRM!")
    print("=" * 60)
    print("Ten system implementuje:")
    print("• 🔧 Dynamiczną Matrycę Reguł (DRM) z matematycznymi wzorami")
    print("• 🎯 Rozszerzony System Uczenia Reguł (RLS) z FRZ")
    print("• 🧠 Sieć neuronową z adaptacyjnym uczeniem")
    print("• 📊 Zaawansowaną analizę i eksport danych")
    print("=" * 60)
    
    print("\nWybierz tryb uruchomienia:")
    print("1. 🚀 Normalny start")
    print("2. 🎭 System demonstracyjny")
    print("3. ⏱️ Test wydajności")
    print("0. 🚪 Wyjście")
    
    try:
        choice = input("\nTwój wybór: ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            demo_system = create_demo_system()
            demo_system.run_advanced_menu()
        elif choice == "3":
            run_benchmark()
        elif choice == "0":
            print("👋 Do zobaczenia!")
        else:
            print("❌ Nieprawidłowy wybór, uruchamiam normalny start...")
            main()
            
    except KeyboardInterrupt:
        print("\n\n👋 Program zakończony przez użytkownika.")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        print("Uruchamiam system w trybie awaryjnym...")
        try:
            system = AdvancedLoopDRMSystem()
            system.run_advanced_menu()
        except:
            print("❌ Nie można uruchomić systemu. Sprawdź instalację bibliotek.")
            print("\n🔧 Wymagane biblioteki:")
            print("  • torch (PyTorch)")
            print("  • numpy")
            print("  • json (wbudowana)")
            print("  • dataclasses (wbudowana)")
            print("  • enum (wbudowana)")
            print("\n📦 Instalacja:")
            print("  pip install torch numpy")
            print("\n🆘 Jeśli problem nadal występuje, sprawdź:")
            print("  • Wersję Pythona (wymagana 3.7+)")
            print("  • Dostępność pamięci RAM")
            print("  • Uprawnienia do zapisu plików")

# Klasy pomocnicze i dodatkowe funkcjonalności

class DRMVisualizer:
    """Klasa do wizualizacji działania DRM"""
    
    def __init__(self, drm_system):
        self.drm = drm_system
        
    def print_rule_evolution(self, history_length=10):
        """Wyświetl ewolucję reguł"""
        print(f"\n📈 EWOLUCJA REGUŁ (ostatnie {history_length} kroków)")
        print("=" * 80)
        
        if len(self.drm.memory) < history_length:
            print("⚠️ Niewystarczająca historia do analizy")
            return
        
        # Analiza zmian siły reguł w czasie
        recent_memory = self.drm.memory[-history_length:]
        
        for i, mem_point in enumerate(recent_memory):
            print(f"\nKrok {mem_point['time']} (FRZ: {mem_point['FRZ']:.3f}):")
            
            # Oblicz siły reguł dla tego momentu
            temp_T = self.drm.T
            self.drm.T = mem_point['time']
            
            rule_strengths = {}
            for rule_id in self.drm.rules.keys():
                strength = self.drm.calculate_rule_strength(rule_id)
                rule_strengths[rule_id] = strength
            
            # Przywróć oryginalny czas
            self.drm.T = temp_T
            
            # Pokaż top 3 reguły
            sorted_rules = sorted(rule_strengths.items(), key=lambda x: x[1], reverse=True)
            for j, (rule_id, strength) in enumerate(sorted_rules[:3]):
                bar_length = int(strength * 20)  # Skala 0-20 znaków
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  {j+1}. {rule_id:<20} [{bar}] {strength:.3f}")
    
    def print_context_analysis(self, context):
        """Analiza kontekstu wejściowego"""
        print(f"\n🔍 ANALIZA KONTEKSTU")
        print("=" * 50)
        
        print(f"Wartości: {[f'{x:.3f}' for x in context]}")
        print(f"Statystyki:")
        print(f"  • Średnia: {np.mean(context):.4f}")
        print(f"  • Odchylenie: {np.std(context):.4f}")
        print(f"  • Minimum: {np.min(context):.4f}")
        print(f"  • Maksimum: {np.max(context):.4f}")
        print(f"  • Zakres: {np.max(context) - np.min(context):.4f}")
        
        # Porównanie z historią
        if len(self.drm.memory) > 0:
            print(f"\nPorównanie z historią:")
            historical_contexts = [mem['context'] for mem in self.drm.memory[-10:]]
            
            if historical_contexts:
                avg_historical = np.mean([np.mean(ctx) for ctx in historical_contexts])
                current_avg = np.mean(context)
                
                difference = abs(current_avg - avg_historical)
                similarity = 1.0 / (1.0 + difference)
                
                print(f"  • Podobieństwo do historii: {similarity:.3f}")
                
                if similarity > 0.8:
                    print("  ✅ Kontekst bardzo podobny do historycznych")
                elif similarity > 0.5:
                    print("  ⚠️ Kontekst umiarkowanie podobny")
                else:
                    print("  🚨 Kontekst znacznie różni się od historycznych")

class PerformanceMonitor:
    """Monitor wydajności systemu"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        self.checkpoints = []
        
    def start_monitoring(self):
        """Rozpocznij monitorowanie"""
        import time
        self.start_time = time.time()
        self.metrics = {
            'total_time': 0,
            'epochs_processed': 0,
            'rules_created': 0,
            'rules_deleted': 0,
            'mode_switches': 0,
            'memory_usage': []
        }
        print("📊 Monitoring wydajności rozpoczęty")
        
    def checkpoint(self, description=""):
        """Utwórz punkt kontrolny"""
        if self.start_time is None:
            return
            
        import time
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        checkpoint_data = {
            'time': elapsed,
            'description': description,
            'timestamp': current_time
        }
        
        self.checkpoints.append(checkpoint_data)
        print(f"⏱️ Checkpoint: {description} ({elapsed:.3f}s)")
        
    def update_metrics(self, **kwargs):
        """Aktualizuj metryki"""
        for key, value in kwargs.items():
            if key in self.metrics:
                if isinstance(self.metrics[key], list):
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = value
                    
    def get_memory_usage(self):
        """Pobierz użycie pamięci (jeśli dostępne)"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            return None
            
    def print_summary(self):
        """Wyświetl podsumowanie wydajności"""
        if self.start_time is None:
            print("❌ Monitoring nie został rozpoczęty")
            return
            
        import time
        total_time = time.time() - self.start_time
        
        print(f"\n📊 PODSUMOWANIE WYDAJNOŚCI")
        print("=" * 50)
        print(f"Całkowity czas: {total_time:.3f}s")
        print(f"Przetworzone epoki: {self.metrics.get('epochs_processed', 0)}")
        
        if self.metrics.get('epochs_processed', 0) > 0:
            avg_time_per_epoch = total_time / self.metrics['epochs_processed']
            print(f"Średni czas na epokę: {avg_time_per_epoch:.4f}s")
            
        print(f"Utworzone reguły: {self.metrics.get('rules_created', 0)}")
        print(f"Usunięte reguły: {self.metrics.get('rules_deleted', 0)}")
        print(f"Zmiany trybu: {self.metrics.get('mode_switches', 0)}")
        
        # Użycie pamięci
        memory_usage = self.get_memory_usage()
        if memory_usage:
            print(f"Użycie pamięci: {memory_usage:.1f} MB")
            
        # Checkpointy
        if self.checkpoints:
            print(f"\nPunkty kontrolne:")
            for i, cp in enumerate(self.checkpoints):
                print(f"  {i+1}. {cp['description']}: {cp['time']:.3f}s")

class ConfigManager:
    """Menedżer konfiguracji systemu"""
    
    @staticmethod
    def save_config(system, filename="config.json"):
        """Zapisz konfigurację systemu"""
        config = {
            'network': {
                'input_size': system.input_size,
                'hidden_size': system.hidden_size,
                'output_size': system.output_size
            },
            'drm': {
                'adaptation_threshold': system.drm.adaptation_threshold,
                'exploration_threshold': system.drm.exploration_threshold,
                'mode': system.drm.mode.value
            },
            'rls': {
                'threshold': system.rls.threshold,
                'window_size': system.rls.window_size
            },
            'optimizer': {
                'lr': system.optimizer.param_groups[0]['lr']
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Konfiguracja zapisana do {filename}")
        except Exception as e:
            print(f"❌ Błąd zapisu konfiguracji: {e}")
            
    @staticmethod
    def load_config(filename="config.json"):
        """Wczytaj konfigurację"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Konfiguracja wczytana z {filename}")
            return config
        except FileNotFoundError:
            print(f"❌ Plik {filename} nie istnieje")
            return None
        except Exception as e:
            print(f"❌ Błąd wczytywania konfiguracji: {e}")
            return None
            
    @staticmethod
    def create_system_from_config(config):
        """Utwórz system na podstawie konfiguracji"""
        if not config:
            return None
            
        try:
            # Utwórz system
            system = AdvancedLoopDRMSystem(
                input_size=config['network']['input_size'],
                hidden_size=config['network']['hidden_size'],
                output_size=config['network']['output_size']
            )
            
            # Zastosuj konfigurację DRM
            system.drm.adaptation_threshold = config['drm']['adaptation_threshold']
            system.drm.exploration_threshold = config['drm']['exploration_threshold']
            system.drm.mode = SystemMode(config['drm']['mode'])
            
            # Zastosuj konfigurację RLS
            system.rls.threshold = config['rls']['threshold']
            system.rls.window_size = config['rls']['window_size']
            
            # Zastosuj konfigurację optymalizatora
            for param_group in system.optimizer.param_groups:
                param_group['lr'] = config['optimizer']['lr']
                
            print("✅ System utworzony na podstawie konfiguracji")
            return system
            
        except Exception as e:
            print(f"❌ Błąd tworzenia systemu z konfiguracji: {e}")
            return None

# Funkcje narzędziowe
def validate_system_integrity(system):
    """Sprawdź integralność systemu"""
    print("\n🔍 SPRAWDZANIE INTEGRALNOŚCI SYSTEMU")
    print("=" * 50)
    
    issues = []
    
    # Sprawdź sieć neuronową
    try:
        test_input = torch.randn(1, system.input_size)
        output = system.model(test_input)
        if output.shape[1] != system.output_size:
            issues.append("Nieprawidłowy rozmiar wyjścia sieci")
    except Exception as e:
        issues.append(f"Błąd sieci neuronowej: {e}")
    
    # Sprawdź DRM
    if len(system.drm.rules) == 0:
        issues.append("Brak reguł w DRM")
        
    for rule_id, rule in system.drm.rules.items():
        if not callable(rule['condition']) or not callable(rule['action']):
            issues.append(f"Nieprawidłowa reguła: {rule_id}")
            
        strength = system.drm.calculate_rule_strength(rule_id)
        if strength < 0:
            issues.append(f"Ujemna siła reguły: {rule_id}")
    
    # Sprawdź RLS
    if system.rls.threshold <= 0:
        issues.append("Nieprawidłowy próg RLS")
        
    if system.rls.window_size <= 0:
        issues.append("Nieprawidłowy rozmiar okna RLS")
    
    # Sprawdź dane treningowe
    if system.training_inputs and system.training_targets:
        if len(system.training_inputs) != len(system.training_targets):
            issues.append("Niezgodność liczby danych wejściowych i docelowych")
            
        for i, (inp, tgt) in enumerate(zip(system.training_inputs, system.training_targets)):
            if len(inp) != system.input_size:
                issues.append(f"Nieprawidłowy rozmiar danych wejściowych #{i}")
            if len(tgt) != system.output_size:
                issues.append(f"Nieprawidłowy rozmiar danych docelowych #{i}")
    
    # Wyświetl wyniki
    if issues:
        print("❌ ZNALEZIONE PROBLEMY:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False
    else:
        print("✅ System jest w pełni sprawny")
        return True

def generate_test_data(input_size, output_size, num_samples=100, pattern="mixed"):
    """Generuj dane testowe z różnymi wzorcami"""
    inputs = []
    targets = []
    
    np.random.seed(42)  # Dla powtarzalności
    
    for i in range(num_samples):
        if pattern == "linear":
            # Wzorzec liniowy
            inp = np.linspace(-2, 2, input_size).tolist()
            tgt = [np.sum(inp[:output_size])]
            while len(tgt) < output_size:
                tgt.append(np.mean(inp))
                
        
        
        elif pattern == "quadratic":
            # Wzorzec kwadratowy
            inp = (np.random.randn(input_size) ** 2).tolist()
            tgt = [np.max(inp[:output_size])]
            while len(tgt) < output_size:
                tgt.append(np.std(inp))
                
        elif pattern == "sinusoidal":
            # Wzorzec sinusoidalny
            x = np.linspace(0, 2*np.pi, input_size)
            inp = np.sin(x + i * 0.1).tolist()
            tgt = [np.mean(inp), np.amplitude(inp) if output_size > 1 else np.mean(inp)]
            while len(tgt) < output_size:
                tgt.append(np.var(inp))
                
        elif pattern == "mixed":
            # Wzorzec mieszany
            if i % 4 == 0:
                inp = np.random.randn(input_size).tolist()
                tgt = [np.mean(inp), np.std(inp)]
            elif i % 4 == 1:
                inp = (np.random.randn(input_size) * 2 + 1).tolist()
                tgt = [np.median(inp), np.max(inp)]
            elif i % 4 == 2:
                inp = np.random.exponential(1, input_size).tolist()
                tgt = [np.min(inp), np.sum(inp)]
            else:
                inp = np.random.uniform(-3, 3, input_size).tolist()
                tgt = [np.var(inp), np.mean(np.abs(inp))]
            
            # Dopasuj rozmiar targets
            while len(tgt) < output_size:
                tgt.append(np.random.randn())
            tgt = tgt[:output_size]
            
        else:  # random
            inp = np.random.randn(input_size).tolist()
            tgt = np.random.randn(output_size).tolist()
        
        inputs.append(inp)
        targets.append(tgt)
    
    return inputs, targets

def run_automated_test():
    """Uruchom zautomatyzowany test systemu"""
    print("🤖 ZAUTOMATYZOWANY TEST SYSTEMU")
    print("=" * 50)
    
    # Utwórz system testowy
    system = AdvancedLoopDRMSystem(input_size=6, hidden_size=12, output_size=3)
    monitor = PerformanceMonitor()
    
    monitor.start_monitoring()
    
    # Sprawdź integralność
    monitor.checkpoint("Sprawdzanie integralności")
    if not validate_system_integrity(system):
        print("❌ Test przerwany - problemy z integralnością")
        return
    
    # Wygeneruj dane testowe
    monitor.checkpoint("Generowanie danych")
    inputs, targets = generate_test_data(6, 3, 80, "mixed")
    system.training_inputs = inputs
    system.training_targets = targets
    
    print(f"✅ Wygenerowano {len(inputs)} próbek danych")
    
    # Trenowanie automatyczne
    monitor.checkpoint("Rozpoczęcie trenowania")
    
    inputs_tensor = torch.FloatTensor(system.training_inputs)
    targets_tensor = torch.FloatTensor(system.training_targets)
    
    epochs = 30
    print(f"🎯 Trenowanie przez {epochs} epok...")
    
    for epoch in range(epochs):
        # Forward pass
        outputs = system.model(inputs_tensor)
        loss = system.criterion(outputs, targets_tensor)
        
        # Oblicz dokładność
        accuracy = system.calculate_prediction_accuracy(outputs, targets_tensor)
        
        # Oblicz FRZ
        FRZ = system.rls.calculate_FRZ(loss.item(), accuracy)
        
        # Backward pass
        system.optimizer.zero_grad()
        loss.backward()
        system.optimizer.step()
        
        # Zastosuj DRM dla próbki danych
        sample_contexts = system.training_inputs[::10]  # Co 10-ta próbka
        for context in sample_contexts:
            system.drm.apply_rules(context, FRZ)
        
        # Wykryj różnice
        difference_detected = system.rls.detect_difference(loss.item())
        
        # Zapisz metryki
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': loss.item(),
            'accuracy': accuracy,
            'FRZ': FRZ,
            'system_mode': system.drm.mode.value,
            'avg_rule_strength': system.drm.calculate_system_average_strength(),
            'active_rules': len(system.drm.rules),
            'difference_detected': difference_detected
        }
        system.training_history.append(epoch_metrics)
        
        # Aktualizuj monitor
        monitor.update_metrics(epochs_processed=epoch + 1)
        
        # Wyświetl postęp co 10 epok
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoka {epoch+1:2d}/{epochs} | "
                  f"Loss: {loss.item():.6f} | "
                  f"FRZ: {FRZ:.3f} | "
                  f"Reguły: {len(system.drm.rules)} | "
                  f"Tryb: {system.drm.mode.value}")
    
    monitor.checkpoint("Zakończenie trenowania")
    
    # Test predykcji
    monitor.checkpoint("Test predykcji")
    test_input = np.random.randn(6).tolist()
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor([test_input])
        prediction = system.model(input_tensor)[0].tolist()
    
    print(f"\n🧪 Test predykcji:")
    print(f"Wejście: {[f'{x:.3f}' for x in test_input]}")
    print(f"Predykcja: {[f'{x:.3f}' for x in prediction]}")
    
    # Analiza DRM
    current_FRZ = system.rls.get_current_FRZ()
    drm_result = system.drm.apply_rules(test_input, current_FRZ)
    
    print(f"\n🔧 Analiza DRM:")
    print(f"Zastosowane reguły: {len(drm_result['applied_rules'])}")
    print(f"Średnia siła systemu: {drm_result['average_strength']:.3f}")
    print(f"Tryb systemu: {drm_result['system_mode']}")
    
    # Podsumowanie
    monitor.checkpoint("Finalizacja testu")
    monitor.print_summary()
    
    # Ocena wyników
    print(f"\n🏆 OCENA WYNIKÓW:")
    
    if system.training_history:
        initial_loss = system.training_history[0]['loss']
        final_loss = system.training_history[-1]['loss']
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"Poprawa loss: {improvement:.1f}%")
        
        if improvement > 50:
            print("✅ Doskonały wynik!")
        elif improvement > 20:
            print("✅ Dobry wynik!")
        elif improvement > 5:
            print("⚠️ Umiarkowany wynik")
        else:
            print("❌ Słaby wynik - system wymaga dostrojenia")
        
        # Analiza stabilności DRM
        mode_changes = 0
        for i in range(1, len(system.training_history)):
            if (system.training_history[i]['system_mode'] != 
                system.training_history[i-1]['system_mode']):
                mode_changes += 1
        
        print(f"Zmiany trybu DRM: {mode_changes}")
        
        if mode_changes > epochs * 0.3:
            print("⚠️ System często zmienia tryby - może być niestabilny")
        elif mode_changes < 2:
            print("⚠️ System rzadko zmienia tryby - może brakować adaptacji")
        else:
            print("✅ Optymalna częstotliwość zmian trybów")
    
    return system

# Funkcje diagnostyczne
def diagnose_training_issues(system):
    """Diagnozuj problemy z trenowaniem"""
    print("\n🔬 DIAGNOZA PROBLEMÓW TRENOWANIA")
    print("=" * 50)
    
    if not system.training_history:
        print("❌ Brak historii trenowania do analizy")
        return
    
    issues = []
    recommendations = []
    
    # Analiza konwergencji
    losses = [h['loss'] for h in system.training_history]
    
    if len(losses) >= 10:
        recent_losses = losses[-10:]
        early_losses = losses[:10]
        
        recent_avg = np.mean(recent_losses)
        early_avg = np.mean(early_losses)
        
        if recent_avg >= early_avg * 0.95:  # Mniej niż 5% poprawy
            issues.append("Brak znaczącej poprawy loss")
            recommendations.append("Zwiększ learning rate lub liczbę epok")
        
        # Sprawdź stabilność
        recent_std = np.std(recent_losses)
        if recent_std > recent_avg * 0.1:  # Odchylenie > 10% średniej
            issues.append("Niestabilne trenowanie")
            recommendations.append("Zmniejsz learning rate")
    
    # Analiza FRZ
    frz_values = [h['FRZ'] for h in system.training_history]
    avg_frz = np.mean(frz_values)
    
    if avg_frz < 0.3:
        issues.append("Niski średni FRZ")
        recommendations.append("Sprawdź jakość danych lub architekturę sieci")
    elif avg_frz > 0.9:
        issues.append("Bardzo wysoki FRZ - możliwe przeuczenie")
        recommendations.append("Dodaj regularyzację lub więcej danych")
    
    # Analiza DRM
    rule_counts = [h['active_rules'] for h in system.training_history]
    
    if max(rule_counts) - min(rule_counts) > len(system.drm.rules) * 0.5:
        issues.append("Duże wahania liczby aktywnych reguł")
        recommendations.append("Dostosuj progi adaptacji DRM")
    
    avg_strength = [h['avg_rule_strength'] for h in system.training_history]
    if np.mean(avg_strength) < 0.3:
        issues.append("Niska średnia siła reguł")
        recommendations.append("Zmniejsz próg adaptacji lub zwiększ próg eksploracji")
    
    # Wyświetl wyniki
    if issues:
        print("🚨 ZIDENTYFIKOWANE PROBLEMY:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n💡 REKOMENDACJE:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("✅ Nie wykryto problemów z trenowaniem")

# Ostatnie funkcje i zakończenie
def create_quick_demo():
    """Szybka demonstracja możliwości systemu"""
    print("⚡ SZYBKA DEMONSTRACJA")
    print("=" * 30)
    
    # Mały system do demonstracji
    system = AdvancedLoopDRMSystem(input_size=4, hidden_size=8, output_size=2)
    
    # Dodaj kilka próbek
    demo_data = [
        ([1, 2, 3, 4], [2.5, 1.2]),
        ([2, 4, 6, 8], [5.0, 2.4]),
        ([-1, -2, -3, -4], [-2.5, 1.2]),
        ([0.5, 1.5, 2.5, 3.5], [2.0, 1.0])
    ]
    
    for inp, tgt in demo_data:
        system.training_inputs.append(inp)
        system.training_targets.append(tgt)
    
    print(f"📊 Dodano {len(demo_data)} próbek demonstracyjnych")
    
    # Krótkie trenowanie
    inputs_tensor = torch.FloatTensor(system.training_inputs)
    targets_tensor = torch.FloatTensor(system.training_targets)
    
    print("🎯 Trenowanie (10 epok)...")
    
    for epoch in range(10):
        outputs = system.model(inputs_tensor)
        loss = system.criterion(outputs, targets_tensor)
        
        system.optimizer.zero_grad()
        loss.backward()
        system.optimizer.step()
        
        # Zastosuj DRM
        FRZ = system.rls.calculate_FRZ(loss.item())
        for context in system.training_inputs:
            system.drm.apply_rules(context, FRZ)
        
        if epoch % 3 == 0:
            print(f"  Epoka {epoch+1}: Loss={loss.item():.4f}, "
                  f"FRZ={FRZ:.3f}, Reguły={len(system.drm.rules)}")
    
    # Test
    test_input = [1.5, 2.5, 3.5, 4.5]
    with torch.no_grad():
        prediction = system.model(torch.FloatTensor([test_input]))[0].tolist()
    
    print(f"\n🧪 Test: {test_input} → {[f'{x:.3f}' for x in prediction]}")
    print("✅ Demonstracja zakończona!")

# Informacje o systemie
def print_system_info():
    """Wyświetl informacje o systemie"""
    print("\n" + "="*60)
    print("    🚀 ZAAWANSOWANY SYSTEM LOOPDRM")
    print("="*60)
    print("📋 OPIS SYSTEMU:")
    print("  • Implementacja Dynamicznej Matrycy Reguł (DRM)")
    print("  • Rozszerzony System Uczenia Reguł (RLS) z FRZ")
    print("  • Adaptacyjne sieci neuronowe z PyTorch")
    print("  • Automatyczna adaptacja trybu eksploracja/eksploatacja")
    print("  • Zaawansowana analiza i wizualizacja")
    
    print(f"\n🔬 WZORY MATEMATYCZNE:")
    print("  • Siła reguły: Si = Wi · log(Ci + 1) · (1 + Ui/T) · Ri")
    print("  • Średnia siła systemu: S̄ = (1/n) Σ Si")
    print("  • FRZ: Function Resonance Zone = 1 - normalized_loss + accuracy_bonus")
    print("  • Różnorodność kontekstu: ||context - avg_context||")
    
    print(f"\n🎛️ PARAMETRY SYSTEMU:")
    print("  • Wi - Waga skuteczności (0.1 - 5.0)")
    print("  • Ci - Zasięg kontekstowy (0.1 - 10.0)")
    print("  • Ui - Użycie reguły (liczba aktywacji)")
    print("  • Ri - Rezonans z pamięcią (0.0 - 1.0)")
    print("  • T - Czas systemowy (liczba iteracji)")
    
    print(f"\n🔧 TRYBY DZIAŁANIA:")
    print("  • EKSPLORACJA - tworzenie nowych reguł, mutacje")
    print("  • EKSPLOATACJA - wykorzystanie najlepszych reguł")
    print("  • Automatyczne przełączanie na podstawie S̄")
    
    print(f"\n📊 MOŻLIWOŚCI:")
    print("  • Trenowanie sieci z adaptacyjnym DRM")
    print("  • Analiza wydajności i diagnostyka")
    print("  • Eksport danych (JSON, CSV, TXT)")
    print("  • Zapis/wczytywanie modeli z metadanymi")
    print("  • Wizualizacja ewolucji reguł")
    print("  • Monitoring wydajności w czasie rzeczywistym")
    
    print(f"\n👨‍💻 AUTOR: System LoopDRM v2.0")
    print(f"📅 WERSJA: Zaawansowana implementacja z matematycznymi wzorami")
    print("="*60)

# Funkcja pomocnicza do debugowania
def debug_system_state(system):
    """Wyświetl szczegółowy stan systemu do debugowania"""
    print("\n🐛 STAN SYSTEMU (DEBUG)")
    print("="*50)
    
    # Stan sieci neuronowej
    print("🧠 SIEĆ NEURONOWA:")
    print(f"  Architektura: {system.input_size} → {system.hidden_size} → {system.output_size}")
    print(f"  Parametry: {sum(p.numel() for p in system.model.parameters())}")
    print(f"  Learning rate: {system.optimizer.param_groups[0]['lr']}")
    
    # Stan DRM
    print(f"\n🔧 DRM:")
    print(f"  Czas systemowy T: {system.drm.T}")
    print(f"  Tryb: {system.drm.mode.value}")
    print(f"  Próg adaptacji: {system.drm.adaptation_threshold}")
    print(f"  Próg eksploracji: {system.drm.exploration_threshold}")
    print(f"  Liczba reguł: {len(system.drm.rules)}")
    print(f"  Rozmiar pamięci: {len(system.drm.memory)}")
    
    # Szczegóły reguł
    if system.drm.rules:
        print(f"\n  Szczegóły reguł:")
        for rule_id, rule in system.drm.rules.items():
            meta = rule['metadata']
            strength = system.drm.calculate_rule_strength(rule_id)
            print(f"    {rule_id}:")
            print(f"      Siła: {strength:.4f}")
            print(f"      W={meta.W:.2f}, C={meta.C:.2f}, U={meta.U:.1f}, R={meta.R:.3f}")
            print(f"      Sukcesy: {meta.success_count}/{meta.total_activations}")
    
    # Stan RLS
    print(f"\n🎯 RLS:")
    print(f"  Próg: {system.rls.threshold}")
    print(f"  Rozmiar okna: {system.rls.window_size}")
    print(f"  Historia: {len(system.rls.history)} punktów")
    print(f"  Wykryte różnice: {len(system.rls.differences)}")
    print(f"  Historia FRZ: {len(system.rls.FRZ_history)} punktów")
    print(f"  Aktualny FRZ: {system.rls.get_current_FRZ():.4f}")
    
    # Dane treningowe
    print(f"\n📊 DANE:")
    print(f"  Próbki treningowe: {len(system.training_inputs)}")
    print(f"  Historia trenowania: {len(system.training_history)} epok")
    
    if system.training_history:
        last_epoch = system.training_history[-1]
        print(f"  Ostatnia epoka:")
        print(f"    Loss: {last_epoch['loss']:.6f}")
        print(f"    FRZ: {last_epoch['FRZ']:.4f}")
        print(f"    Accuracy: {last_epoch.get('accuracy', 'N/A')}")
    
    # Pamięć systemowa
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"\n💾 PAMIĘĆ:")
        print(f"  Użycie RAM: {memory_mb:.1f} MB")
    except ImportError:
        print(f"\n💾 PAMIĘĆ: Moduł psutil niedostępny")

# Funkcja do testowania konkretnych scenariuszy
def run_scenario_test(scenario="basic"):
    """Uruchom test konkretnego scenariusza"""
    print(f"\n🎬 TEST SCENARIUSZA: {scenario.upper()}")
    print("="*50)
    
    if scenario == "basic":
        # Podstawowy test funkcjonalności
        system = AdvancedLoopDRMSystem(4, 8, 2)
        inputs, targets = generate_test_data(4, 2, 20, "linear")
        system.training_inputs = inputs
        system.training_targets = targets
        
        # Krótkie trenowanie
        system._train_epochs(10)
        print("✅ Podstawowy scenariusz zakończony")
        
    elif scenario == "stress":
        # Test obciążeniowy
        print("⚡ Test obciążeniowy - duże dane")
        system = AdvancedLoopDRMSystem(20, 40, 10)
        inputs, targets = generate_test_data(20, 10, 500, "mixed")
        system.training_inputs = inputs
        system.training_targets = targets
        
        import time
        start_time = time.time()
        system._train_epochs(50)
        end_time = time.time()
        
        print(f"⏱️ Czas trenowania: {end_time - start_time:.2f}s")
        print("✅ Test obciążeniowy zakończony")
        
    elif scenario == "adaptation":
        # Test adaptacji DRM
        print("🔄 Test adaptacji DRM")
        system = AdvancedLoopDRMSystem(6, 12, 3)
        
        # Różne fazy danych
        phase1_inputs, phase1_targets = generate_test_data(6, 3, 30, "linear")
        phase2_inputs, phase2_targets = generate_test_data(6, 3, 30, "quadratic")
        
        # Faza 1
        system.training_inputs = phase1_inputs
        system.training_targets = phase1_targets
        system._train_epochs(20)
        
        print("📊 Zmiana wzorca danych...")
        
        # Faza 2
        system.training_inputs.extend(phase2_inputs)
        system.training_targets.extend(phase2_targets)
        system._train_epochs(20)
        
        print("✅ Test adaptacji zakończony")
        
    else:
        print(f"❌ Nieznany scenariusz: {scenario}")

# Dodatkowa metoda do klasy AdvancedLoopDRMSystem
def _train_epochs(self, epochs):
    """Pomocnicza metoda do trenowania określonej liczby epok"""
    if not self.training_inputs or not self.training_targets:
        print("❌ Brak danych treningowych")
        return
    
    inputs_tensor = torch.FloatTensor(self.training_inputs)
    targets_tensor = torch.FloatTensor(self.training_targets)
    
    for epoch in range(epochs):
        outputs = self.model(inputs_tensor)
        loss = self.criterion(outputs, targets_tensor)
        
        accuracy = self.calculate_prediction_accuracy(outputs, targets_tensor)
        FRZ = self.rls.calculate_FRZ(loss.item(), accuracy)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Zastosuj DRM dla próbki kontekstów
        sample_size = min(10, len(self.training_inputs))
        for i in range(0, len(self.training_inputs), len(self.training_inputs)//sample_size):
            self.drm.apply_rules(self.training_inputs[i], FRZ)
        
        difference_detected = self.rls.detect_difference(loss.item())
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': loss.item(),
            'accuracy': accuracy,
            'FRZ': FRZ,
            'system_mode': self.drm.mode.value,
            'avg_rule_strength': self.drm.calculate_system_average_strength(),
            'active_rules': len(self.drm.rules),
            'difference_detected': difference_detected
        }
        self.training_history.append(epoch_metrics)

# Dodaj metodę do klasy AdvancedLoopDRMSystem
AdvancedLoopDRMSystem._train_epochs = _train_epochs

# Finalne menu rozszerzone
def show_extended_menu():
    """Pokaż rozszerzone menu opcji"""
    print("\n" + "="*60)
    print("           🚀 ZAAWANSOWANY SYSTEM LOOPDRM")
    print("="*60)
    print("📋 MENU GŁÓWNE:")
    print("  1. 🎯 Dodaj dane treningowe")
    print("  2. 🧠 Trenuj sieć neuronową")
    print("  3. 🧪 Testuj model")
    print("  4. 💾 Zapisz model")
    print("  5. 📂 Wczytaj model")
    print("  6. ⚙️  Konfiguracja systemu")
    print("  7. 📊 Analiza wydajności")
    print("  8. 📤 Eksport danych")
    print()
    print("🔧 NARZĘDZIA ZAAWANSOWANE:")
    print("  9. 🔍 Szczegółowe statystyki DRM")
    print(" 10. 🎬 Testy scenariuszy")
    print(" 11. ⚡ Szybka demonstracja")
    print(" 12. 🤖 Test automatyczny")
    print(" 13. 🔬 Diagnoza problemów")
    print(" 14. 🐛 Stan systemu (debug)")
    print(" 15. ℹ️  Informacje o systemie")
    print()
    print("  0. 🚪 Wyjście")
    print("="*60)

# Rozszerzona metoda run_advanced_menu
def run_extended_advanced_menu(self):
    """Uruchom rozszerzone menu zaawansowane"""
    while True:
        show_extended_menu()
        
        try:
            choice = input("\n🎯 Wybierz opcję: ").strip()
            
            if choice == "1":
                self.add_training_data()
            elif choice == "2":
                self.train_network()
            elif choice == "3":
                self.test_model()
            elif choice == "4":
                self.save_model()
            elif choice == "5":
                self.load_model()
            elif choice == "6":
                self.configure_system()
            elif choice == "7":
                self.analyze_performance()
            elif choice == "8":
                self.export_data()
            elif choice == "9":
                stats = self.drm.get_detailed_statistics()
                print("\n📊 SZCZEGÓŁOWE STATYSTYKI DRM:")
                print(json.dumps(stats, indent=2, ensure_ascii=False))
            elif choice == "10":
                print("\nDostępne scenariusze:")
                print("1. basic - Podstawowy test")
                print("2. stress - Test obciążeniowy")
                print("3. adaptation - Test adaptacji")
                scenario_choice = input("Wybierz scenariusz: ").strip()
                scenarios = {"1": "basic", "2": "stress", "3": "adaptation"}
                run_scenario_test(scenarios.get(scenario_choice, "basic"))
            elif choice == "11":
                create_quick_demo()
            elif choice == "12":
                run_automated_test()
            elif choice == "13":
                diagnose_training_issues(self)
            elif choice == "14":
                debug_system_state(self)
            elif choice == "15":
                print_system_info()
            elif choice == "0":
                print("👋 Dziękujemy za korzystanie z systemu LoopDRM!")
                break
            else:
                print("❌ Nieprawidłowy wybór. Spróbuj ponownie.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Przerwano przez użytkownika.")
            break
        except Exception as e:
            print(f"\n❌ Błąd: {e}")
            print("Kontynuowanie działania...")

# Zastąp oryginalną metodę
AdvancedLoopDRMSystem.run_advanced_menu = run_extended_advanced_menu

# Końcowe informacje
print("\n" + "🎉" * 20)
print("   SYSTEM LOOPDRM GOTOWY DO UŻYCIA!")
print("🎉" * 20)
print("\n📚 Aby rozpocząć, uruchom:")
print("   python loopdrm.py")
print("\n🔗 Lub użyj w kodzie:")
print("   from loopdrm import AdvancedLoopDRMSystem")
print("   system = AdvancedLoopDRMSystem()")
print("   system.run_advanced_menu()")
print("\n✨ Powodzenia w eksperymentach z zaawansowanym systemem LoopDRM!")
print("🚀 System zawiera pełną implementację matematycznych wzorów DRM")
print("🎯 Gotowy do badań nad adaptacyjnymi systemami uczenia maszynowego")
print("\n" + "="*60)


