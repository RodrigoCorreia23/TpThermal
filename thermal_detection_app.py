import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class ThermalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção Térmica de Equipamentos Elétricos")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f8ff')
        
        # Variáveis
        self.current_image = None
        self.original_image = None
        self.detection_results = []
        self.red_threshold = tk.IntVar(value=180)
        self.white_threshold = tk.IntVar(value=240)
        self.min_area = tk.IntVar(value=100)
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Notebook para abas
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Aba de Detecção
        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text="🔍 Detecção")
        self.setup_detection_tab(detection_frame)
        
        # Aba de Treinamento
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="🧠 Treinamento IA")
        self.setup_training_tab(training_frame)
        
        # Aba de Configurações
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="⚙️ Configurações")
        self.setup_settings_tab(settings_frame)
        
    def setup_detection_tab(self, parent):
        # Frame principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Frame superior - controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding=10)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Botões de controle
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="📁 Carregar Imagem", 
                  command=self.load_image).pack(side='left', padx=(0, 10))
        
        self.process_btn = ttk.Button(btn_frame, text="🔍 Detectar Zonas Críticas", 
                                     command=self.process_image, state='disabled')
        self.process_btn.pack(side='left', padx=(0, 10))
        
        ttk.Button(btn_frame, text="💾 Exportar Resultados", 
                  command=self.export_results).pack(side='left', padx=(0, 10))
        
        ttk.Button(btn_frame, text="📊 Gerar Relatório", 
                  command=self.generate_report).pack(side='left')
        
        # Frame do meio - imagem e resultados
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill='both', expand=True)
        
        # Frame da imagem
        image_frame = ttk.LabelFrame(middle_frame, text="Imagem Térmica", padding=10)
        image_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Canvas para imagem
        self.image_canvas = tk.Canvas(image_frame, bg='white', width=600, height=400)
        self.image_canvas.pack(fill='both', expand=True)
        
        # Frame de resultados
        results_frame = ttk.LabelFrame(middle_frame, text="Resultados da Detecção", padding=10)
        results_frame.pack(side='right', fill='y')
        results_frame.config(width=300)
        
        # Treeview para resultados
        columns = ('ID', 'Temp', 'Severidade', 'Confiança')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=70)
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Frame de estatísticas
        stats_frame = ttk.LabelFrame(results_frame, text="Estatísticas", padding=10)
        stats_frame.pack(fill='x', pady=(10, 0))
        
        self.stats_labels = {}
        stats_info = [
            ('total', 'Total de Detecções:'),
            ('critical', 'Zonas Críticas:'),
            ('warning', 'Zonas de Atenção:'),
            ('max_temp', 'Temp. Máxima:')
        ]
        
        for key, text in stats_info:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=text).pack(side='left')
            self.stats_labels[key] = ttk.Label(frame, text="0", font=('Arial', 10, 'bold'))
            self.stats_labels[key].pack(side='right')
    
    def setup_training_tab(self, parent):
        # Frame principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="Processo de Treinamento da IA", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Notebook para seções do treinamento
        training_notebook = ttk.Notebook(main_frame)
        training_notebook.pack(fill='both', expand=True)
        
        # Seção 1: Preparação dos Dados
        data_frame = ttk.Frame(training_notebook)
        training_notebook.add(data_frame, text="📊 Dados")
        
        data_text = """
PREPARAÇÃO DOS DADOS PARA TREINAMENTO

1. Coleta de Imagens:
   • Colete 1000+ imagens termográficas FLIR
   • Inclua diferentes condições (dia/noite, temperaturas)
   • Varie tipos de equipamentos (postes, transformadores, cabos)
   • Resolução mínima: 640x480 pixels

2. Anotação Manual:
   • Use ferramentas como LabelImg ou CVAT
   • Anote zonas vermelhas como "zona_critica"
   • Anote zonas brancas como "zona_critica_extrema"
   • Anote zonas amarelas como "zona_atencao"
   • Formato: YOLO (.txt) ou COCO (.json)

3. Divisão do Dataset:
   • 70% para treinamento (700+ imagens)
   • 20% para validação (200+ imagens)
   • 10% para teste (100+ imagens)

4. Estrutura de Pastas:
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
        """
        
        text_widget = tk.Text(data_frame, wrap='word', font=('Consolas', 10))
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert('1.0', data_text)
        text_widget.config(state='disabled')
        
        # Seção 2: Código de Treinamento
        code_frame = ttk.Frame(training_notebook)
        training_notebook.add(code_frame, text="💻 Código")
        
        code_text = '''
# CÓDIGO COMPLETO PARA TREINAMENTO COM YOLOV8

import os
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class ThermalDetectionTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        
    def prepare_dataset_yaml(self):
        """Cria arquivo de configuração do dataset"""
        yaml_content = f"""
path: {self.dataset_path}
train: images/train
val: images/val
test: images/test

nc: 3  # número de classes
names: ['zona_atencao', 'zona_critica', 'zona_critica_extrema']
        """
        
        with open('thermal_dataset.yaml', 'w') as f:
            f.write(yaml_content)
            
    def train_model(self):
        """Treina o modelo YOLOv8"""
        # Carregar modelo pré-treinado
        self.model = YOLO('yolov8n.pt')
        
        # Configurações de treinamento
        results = self.model.train(
            data='thermal_dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name='thermal_detection',
            patience=10,
            save=True,
            plots=True,
            device='cpu',  # use 'cuda' se tiver GPU
            workers=4,
            lr0=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,
            cls=0.5,
            dfl=1.5
        )
        
        return results
        
    def validate_model(self):
        """Valida o modelo treinado"""
        if self.model is None:
            self.model = YOLO('runs/detect/thermal_detection/weights/best.pt')
            
        results = self.model.val(data='thermal_dataset.yaml')
        return results
        
    def detect_thermal_zones(self, image_path, conf_threshold=0.5):
        """Detecta zonas térmicas em uma imagem"""
        if self.model is None:
            self.model = YOLO('runs/detect/thermal_detection/weights/best.pt')
            
        results = self.model(image_path, conf=conf_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2-x1, y2-y1],
                        'confidence': conf,
                        'class': cls,
                        'class_name': ['zona_atencao', 'zona_critica', 'zona_critica_extrema'][cls]
                    })
                    
        return detections

# EXEMPLO DE USO
if __name__ == "__main__":
    # Inicializar treinador
    trainer = ThermalDetectionTrainer('path/to/your/dataset')
    
    # Preparar dataset
    trainer.prepare_dataset_yaml()
    
    # Treinar modelo
    print("Iniciando treinamento...")
    results = trainer.train_model()
    
    # Validar modelo
    print("Validando modelo...")
    val_results = trainer.validate_model()
    
    # Testar detecção
    detections = trainer.detect_thermal_zones('test_image.jpg')
    print(f"Detectadas {len(detections)} zonas térmicas")

# MÉTRICAS DE AVALIAÇÃO
def calculate_metrics(true_boxes, pred_boxes, iou_threshold=0.5):
    """Calcula métricas de precisão, recall e mAP"""
    
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    tp = 0
    fp = 0
    fn = len(true_boxes)
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_idx = -1
        
        for i, true_box in enumerate(true_boxes):
            iou = calculate_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
                
        if best_iou >= iou_threshold:
            tp += 1
            fn -= 1
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
        '''
        
        code_widget = tk.Text(code_frame, wrap='word', font=('Consolas', 9))
        code_widget.pack(fill='both', expand=True, padx=10, pady=10)
        code_widget.insert('1.0', code_text)
        code_widget.config(state='disabled')
        
        # Seção 3: Métricas
        metrics_frame = ttk.Frame(training_notebook)
        training_notebook.add(metrics_frame, text="📈 Métricas")
        
        # Criar gráfico de métricas exemplo
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        
        # Gráfico 1: Loss durante treinamento
        epochs = range(1, 101)
        train_loss = [0.8 - 0.007*x + 0.001*np.sin(x/5) for x in epochs]
        val_loss = [0.85 - 0.006*x + 0.002*np.sin(x/3) for x in epochs]
        
        ax1.plot(epochs, train_loss, label='Train Loss', color='blue')
        ax1.plot(epochs, val_loss, label='Val Loss', color='red')
        ax1.set_title('Loss Durante Treinamento')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Gráfico 2: mAP
        map_scores = [0.3 + 0.006*x - 0.00002*x**2 for x in epochs]
        ax2.plot(epochs, map_scores, color='green')
        ax2.set_title('mAP@0.5 Durante Treinamento')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('mAP')
        ax2.grid(True)
        
        # Gráfico 3: Precisão e Recall
        precision = [0.4 + 0.005*x - 0.00001*x**2 for x in epochs]
        recall = [0.35 + 0.0055*x - 0.00001*x**2 for x in epochs]
        
        ax3.plot(epochs, precision, label='Precisão', color='purple')
        ax3.plot(epochs, recall, label='Recall', color='orange')
        ax3.set_title('Precisão e Recall')
        ax3.set_xlabel('Épocas')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)
        
        # Gráfico 4: Matriz de Confusão
        confusion_matrix = np.array([[85, 10, 5], [8, 92, 0], [3, 2, 95]])
        im = ax4.imshow(confusion_matrix, cmap='Blues')
        ax4.set_title('Matriz de Confusão')
        ax4.set_xlabel('Predito')
        ax4.set_ylabel('Real')
        
        classes = ['Atenção', 'Crítico', 'Extremo']
        ax4.set_xticks(range(3))
        ax4.set_yticks(range(3))
        ax4.set_xticklabels(classes)
        ax4.set_yticklabels(classes)
        
        # Adicionar valores na matriz
        for i in range(3):
            for j in range(3):
                ax4.text(j, i, confusion_matrix[i, j], ha='center', va='center')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, metrics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def setup_settings_tab(self, parent):
        # Frame principal
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurações de Detecção
        detection_frame = ttk.LabelFrame(main_frame, text="Configurações de Detecção", padding=20)
        detection_frame.pack(fill='x', pady=(0, 20))
        
        # Limiar Vermelho
        red_frame = ttk.Frame(detection_frame)
        red_frame.pack(fill='x', pady=10)
        ttk.Label(red_frame, text="Limiar para Zona Vermelha (RGB):").pack(anchor='w')
        red_scale = ttk.Scale(red_frame, from_=100, to=255, variable=self.red_threshold, orient='horizontal')
        red_scale.pack(fill='x', pady=5)
        ttk.Label(red_frame, textvariable=self.red_threshold).pack(anchor='e')
        
        # Limiar Branco
        white_frame = ttk.Frame(detection_frame)
        white_frame.pack(fill='x', pady=10)
        ttk.Label(white_frame, text="Limiar para Zona Branca (RGB):").pack(anchor='w')
        white_scale = ttk.Scale(white_frame, from_=200, to=255, variable=self.white_threshold, orient='horizontal')
        white_scale.pack(fill='x', pady=5)
        ttk.Label(white_frame, textvariable=self.white_threshold).pack(anchor='e')
        
        # Área Mínima
        area_frame = ttk.Frame(detection_frame)
        area_frame.pack(fill='x', pady=10)
        ttk.Label(area_frame, text="Área Mínima (pixels):").pack(anchor='w')
        area_scale = ttk.Scale(area_frame, from_=50, to=1000, variable=self.min_area, orient='horizontal')
        area_scale.pack(fill='x', pady=5)
        ttk.Label(area_frame, textvariable=self.min_area).pack(anchor='e')
        
        # Confiança
        conf_frame = ttk.Frame(detection_frame)
        conf_frame.pack(fill='x', pady=10)
        ttk.Label(conf_frame, text="Limiar de Confiança:").pack(anchor='w')
        conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, variable=self.confidence_threshold, orient='horizontal')
        conf_scale.pack(fill='x', pady=5)
        ttk.Label(conf_frame, textvariable=self.confidence_threshold).pack(anchor='e')
        
        # Dicas
        tips_frame = ttk.LabelFrame(main_frame, text="Dicas de Configuração", padding=20)
        tips_frame.pack(fill='x')
        
        tips_text = """
DICAS PARA OTIMIZAR A DETECÇÃO:

🌡️ Temperaturas de Referência:
   • Zonas Normais: 20-35°C (azul/verde)
   • Zonas de Atenção: 35-45°C (amarelo)
   • Zonas Críticas: 45-60°C (vermelho)
   • Zonas Extremas: >60°C (branco)

🎯 Configuração dos Limiares:
   • RGB Vermelho: 180-220 (ajuste conforme iluminação)
   • RGB Branco: 240-255 (zonas mais quentes)
   • Área Mínima: 100-500px (evita ruído)
   • Confiança: 0.5-0.8 (balance precisão/recall)

🔧 Condições Ambientais:
   • Temperatura ambiente afeta a escala térmica
   • Ajuste limiares conforme estação do ano
   • Considere reflexos e interferências
   • Calibre regularmente o equipamento FLIR

⚡ Tipos de Equipamentos:
   • Transformadores: atenção especial a conexões
   • Cabos: pontos de emenda são críticos
   • Isoladores: degradação causa aquecimento
   • Chaves: contatos podem superaquecer
        """
        
        tips_widget = tk.Text(tips_frame, wrap='word', height=15, font=('Arial', 10))
        tips_widget.pack(fill='both', expand=True)
        tips_widget.insert('1.0', tips_text)
        tips_widget.config(state='disabled')
    
    def load_image(self):
        """Carrega uma imagem térmica e aplica ROI"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem Térmica",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if file_path:
            try:
                # 1) Carrega com OpenCV
                img = cv2.imread(file_path)

                # 2) Aplica recorte para ignorar os primeiros 50px de topo e esquerda
                roi = img[50:, 50:]

                # 3) Armazena na sua aplicação
                self.original_image = roi
                self.current_image = self.original_image.copy()

                # 4) Exibe e habilita o processamento
                self.display_image(self.current_image)
                self.process_btn.config(state='normal')

                # Limpa resultados anteriores
                self.detection_results = []
                self.update_results_display()

                messagebox.showinfo("Sucesso", "Imagem carregada e recortada com sucesso!")

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")


    
    def display_image(self, image):
        """Exibe a imagem no canvas"""
        if image is None:
            return
            
        # Redimensionar para caber no canvas
        height, width = image.shape[:2]
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            scale = min(canvas_width/width, canvas_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height))
            
            # Converter BGR para RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Limpar canvas e exibir imagem
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width//2, canvas_height//2, 
                image=photo, anchor='center'
            )
            self.image_canvas.image = photo  # Manter referência
    
    def process_image(self):
        """Processa a imagem para detectar zonas térmicas críticas"""
        if self.original_image is None:
            return
            
        def process_thread():
            try:
                # Simular processamento (substitua por modelo real)
                import time
                time.sleep(2)  # Simular tempo de processamento
                
                # Detectar zonas críticas usando processamento de imagem básico
                detections = self.detect_thermal_zones_basic(self.original_image)
                
                # Atualizar interface na thread principal
                self.root.after(0, lambda: self.process_complete(detections))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro no processamento: {str(e)}"))
        
        # Executar em thread separada
        threading.Thread(target=process_thread, daemon=True).start()
        
        # Mostrar progresso
        self.process_btn.config(text="Processando...", state='disabled')
    
    def detect_thermal_zones_basic(self, image):
        """Detecção básica usando processamento de imagem (substitua por modelo IA)"""
        detections = []
        
        # Converter para HSV para melhor detecção de cores
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir ranges para cores críticas
        # Vermelho (zona crítica)
        red_lower1 = np.array([0, 50, self.red_threshold.get()])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, self.red_threshold.get()])
        red_upper2 = np.array([180, 255, 255])
        
        # Branco/Amarelo muito claro (zona extrema)
        white_lower = np.array([0, 0, self.white_threshold.get()])
        white_upper = np.array([180, 30, 255])
        
        # Criar máscaras
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Processar máscaras
        for mask, severity, temp_base in [(red_mask, "critical", 50), (white_mask, "extreme", 65)]:
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area.get():
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Simular temperatura e confiança
                    temperature = temp_base + np.random.uniform(-5, 15)
                    confidence = min(0.95, area / 1000 + np.random.uniform(0.1, 0.3))
                    
                    if confidence >= self.confidence_threshold.get():
                        detections.append({
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'confidence': confidence,
                            'temperature': temperature,
                            'severity': severity,
                            'area': area
                        })
        
        return detections
    
    def process_complete(self, detections):
        """Chamado quando o processamento é concluído"""
        self.detection_results = detections
        
        # Desenhar detecções na imagem
        result_image = self.original_image.copy()
        
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            color = (0, 0, 255) if detection['severity'] == 'critical' else (0, 255, 255)  # Vermelho ou Amarelo
            
            # Desenhar retângulo
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Desenhar label
            label = f"{detection['temperature']:.1f}°C ({detection['confidence']:.2f})"
            cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Atualizar exibição
        self.current_image = result_image
        self.display_image(self.current_image)
        self.update_results_display()
        
        # Restaurar botão
        self.process_btn.config(text="🔍 Detectar Zonas Críticas", state='normal')
        
        messagebox.showinfo("Concluído", f"Processamento concluído! {len(detections)} zonas detectadas.")
    
    def update_results_display(self):
        """Atualiza a exibição dos resultados"""
        # Limpar árvore
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Adicionar resultados
        for i, detection in enumerate(self.detection_results):
            self.results_tree.insert('', 'end', values=(
                i + 1,
                f"{detection['temperature']:.1f}°C",
                "Crítico" if detection['severity'] == 'critical' else "Extremo",
                f"{detection['confidence']:.2f}"
            ))
        
        # Atualizar estatísticas
        total = len(self.detection_results)
        critical = len([d for d in self.detection_results if d['severity'] == 'critical'])
        extreme = len([d for d in self.detection_results if d['severity'] == 'extreme'])
        max_temp = max([d['temperature'] for d in self.detection_results]) if self.detection_results else 0
        
        self.stats_labels['total'].config(text=str(total))
        self.stats_labels['critical'].config(text=str(critical), foreground='red')
        self.stats_labels['warning'].config(text=str(extreme), foreground='orange')
        self.stats_labels['max_temp'].config(text=f"{max_temp:.1f}°C", foreground='red' if max_temp > 60 else 'black')
    
    def export_results(self):
        """Exporta os resultados para JSON"""
        if not self.detection_results:
            messagebox.showwarning("Aviso", "Nenhum resultado para exportar!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Resultados",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Todos os arquivos", "*.*")]
        )
        
        if file_path:
            try:
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'total_detections': len(self.detection_results),
                    'settings': {
                        'red_threshold': self.red_threshold.get(),
                        'white_threshold': self.white_threshold.get(),
                        'min_area': self.min_area.get(),
                        'confidence_threshold': self.confidence_threshold.get()
                    },
                    'detections': self.detection_results,
                    'statistics': {
                        'critical_zones': len([d for d in self.detection_results if d['severity'] == 'critical']),
                        'extreme_zones': len([d for d in self.detection_results if d['severity'] == 'extreme']),
                        'max_temperature': max([d['temperature'] for d in self.detection_results]) if self.detection_results else 0,
                        'avg_confidence': np.mean([d['confidence'] for d in self.detection_results]) if self.detection_results else 0
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Sucesso", f"Resultados exportados para {file_path}")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar: {str(e)}")
    
    def generate_report(self):
        """Gera relatório detalhado em HTML"""
        if not self.detection_results:
            messagebox.showwarning("Aviso", "Nenhum resultado para gerar relatório!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Relatório",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("Todos os arquivos", "*.*")]
        )
        
        if file_path:
            try:
                html_content = self.create_html_report()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                messagebox.showinfo("Sucesso", f"Relatório gerado: {file_path}")
                
                # Abrir no navegador
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao gerar relatório: {str(e)}")
    
    def create_html_report(self):
        """Cria conteúdo HTML do relatório"""
        critical_count = len([d for d in self.detection_results if d['severity'] == 'critical'])
        extreme_count = len([d for d in self.detection_results if d['severity'] == 'extreme'])
        max_temp = max([d['temperature'] for d in self.detection_results]) if self.detection_results else 0
        avg_temp = np.mean([d['temperature'] for d in self.detection_results]) if self.detection_results else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Detecção Térmica</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .critical {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }}
        .warning {{ background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); }}
        .table-container {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .critical-row {{ background-color: #ffebee; }}
        .extreme-row {{ background-color: #fff3e0; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌡️ Relatório de Detecção Térmica</h1>
            <p>Análise de Equipamentos Elétricos - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(self.detection_results)}</div>
                <div class="stat-label">Total de Detecções</div>
            </div>
            <div class="stat-card critical">
                <div class="stat-value">{critical_count}</div>
                <div class="stat-label">Zonas Críticas</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{extreme_count}</div>
                <div class="stat-label">Zonas Extremas</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max_temp:.1f}°C</div>
                <div class="stat-label">Temperatura Máxima</div>
            </div>
        </div>
        
        <h2>📊 Detalhes das Detecções</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Posição (X, Y)</th>
                        <th>Dimensões (W × H)</th>
                        <th>Temperatura</th>
                        <th>Severidade</th>
                        <th>Confiança</th>
                        <th>Área (px²)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, detection in enumerate(self.detection_results):
            row_class = "critical-row" if detection['severity'] == 'critical' else "extreme-row"
            severity_text = "🔴 Crítica" if detection['severity'] == 'critical' else "⚪ Extrema"
            
            html += f"""
                    <tr class="{row_class}">
                        <td>{i + 1}</td>
                        <td>({detection['x']}, {detection['y']})</td>
                        <td>{detection['width']} × {detection['height']}</td>
                        <td><strong>{detection['temperature']:.1f}°C</strong></td>
                        <td>{severity_text}</td>
                        <td>{detection['confidence']:.2f}</td>
                        <td>{detection['area']}</td>
                    </tr>
            """
        
        html += f"""
                </tbody>
            </table>
        </div>
        
        <h2>⚙️ Configurações Utilizadas</h2>
        <ul>
            <li><strong>Limiar Vermelho:</strong> {self.red_threshold.get()}</li>
            <li><strong>Limiar Branco:</strong> {self.white_threshold.get()}</li>
            <li><strong>Área Mínima:</strong> {self.min_area.get()} pixels</li>
            <li><strong>Confiança Mínima:</strong> {self.confidence_threshold.get():.2f}</li>
        </ul>
        
        <h2>📈 Análise e Recomendações</h2>
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 20px;">
            <h3>Status Geral:</h3>
        """
        
        if critical_count > 0 or extreme_count > 0:
            html += f"""
            <p style="color: #dc3545;"><strong>⚠️ ATENÇÃO NECESSÁRIA</strong></p>
            <p>Foram detectadas {critical_count + extreme_count} zonas que requerem atenção imediata.</p>
            """
        else:
            html += f"""
            <p style="color: #28a745;"><strong>✅ SITUAÇÃO NORMAL</strong></p>
            <p>Nenhuma zona crítica detectada. Equipamento operando dentro dos parâmetros normais.</p>
            """
        
        html += f"""
            <h3>Recomendações:</h3>
            <ul>
        """
        
        if max_temp > 60:
            html += "<li>🔥 <strong>Inspeção urgente necessária</strong> - Temperatura acima de 60°C detectada</li>"
        if critical_count > 2:
            html += "<li>🔧 <strong>Manutenção preventiva recomendada</strong> - Múltiplas zonas críticas</li>"
        if extreme_count > 0:
            html += "<li>⚡ <strong>Verificar conexões elétricas</strong> - Zonas extremas podem indicar falhas</li>"
        
        html += """
            <li>📅 Realizar nova inspeção em 30 dias</li>
            <li>📋 Manter registro histórico das temperaturas</li>
            <li>🛠️ Considerar upgrade de equipamentos antigos</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Relatório gerado automaticamente pelo Sistema de Detecção Térmica</p>
            <p>Para mais informações, consulte o manual técnico do equipamento</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html

def main():
    root = tk.Tk()
    app = ThermalDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()