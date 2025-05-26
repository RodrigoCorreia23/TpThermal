"""
Analisador Térmico REAL - Usa modelo YOLO treinado
Detecta pontos críticos reais em novas imagens termográficas
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
import threading
import glob

# Verificar YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
    print("✅ YOLO disponível")
except ImportError:
    HAS_YOLO = False
    print("❌ YOLO não encontrado")

class RealThermalDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 Detector Térmico REAL - Análise com IA Treinada")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f8ff')
        
        # Modelo YOLO
        self.model = None
        self.model_path = None
        self.model_loaded = False
        
        # Imagens
        self.current_image = None
        self.original_image = None
        self.current_image_path = None
        self.detection_results = []
        
        # Configurações do modelo
        self.confidence_threshold = tk.DoubleVar(value=0.25)  # Mais baixo para detectar mais
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.img_size = tk.IntVar(value=640)
        
        # Mapeamento das classes (ajuste conforme seu modelo)
        self.class_mapping = {
            0: {
                "name": "zona_atencao", 
                "display": "Zona de Atenção",
                "color": (0, 255, 255),  # Amarelo em BGR
                "severity": "ATENÇÃO",
                "temp_est": "35-45°C",
                "priority": 3
            },
            1: {
                "name": "zona_critica", 
                "display": "Zona Crítica",
                "color": (0, 0, 255),    # Vermelho em BGR
                "severity": "CRÍTICO", 
                "temp_est": "45-60°C",
                "priority": 2
            },
            2: {
                "name": "zona_extrema", 
                "display": "Zona Extrema",
                "color": (255, 255, 255), # Branco em BGR
                "severity": "EXTREMO",
                "temp_est": ">60°C", 
                "priority": 1
            }
        }
        
        self.setup_ui()
        self.load_model_automatically()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Cabeçalho
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text="🔥 Detector Térmico REAL com IA", 
                               font=('Arial', 20, 'bold'))
        title_label.pack()
        
        self.model_status = ttk.Label(header_frame, text="🔴 Carregando modelo...", 
                                     font=('Arial', 12))
        self.model_status.pack()
        
        # Controles
        controls_frame = ttk.LabelFrame(main_frame, text="🎛️ Controles", padding=15)
        controls_frame.pack(fill='x', pady=(0, 15))
        
        # Linha 1 - Botões principais
        btn_row1 = ttk.Frame(controls_frame)
        btn_row1.pack(fill='x', pady=(0, 10))
        
        ttk.Button(btn_row1, text="📁 Nova Imagem", 
                  command=self.load_new_image, width=15).pack(side='left', padx=(0, 10))
        
        self.detect_btn = ttk.Button(btn_row1, text="🔍 DETECTAR PONTOS", 
                                    command=self.detect_critical_points, 
                                    width=20, state='disabled')
        self.detect_btn.pack(side='left', padx=(0, 10))
        
        ttk.Button(btn_row1, text="📂 Analisar Pasta", 
                  command=self.batch_analyze, width=15).pack(side='left', padx=(0, 10))
        
        ttk.Button(btn_row1, text="💾 Salvar Resultados", 
                  command=self.save_results, width=15).pack(side='left')
        
        # Linha 2 - Configurações
        config_row = ttk.Frame(controls_frame)
        config_row.pack(fill='x')
        
        ttk.Label(config_row, text="Sensibilidade:").pack(side='left', padx=(0, 5))
        
        conf_scale = ttk.Scale(config_row, from_=0.1, to=0.8, variable=self.confidence_threshold, 
                              orient='horizontal', length=200)
        conf_scale.pack(side='left', padx=(0, 10))
        
        self.conf_display = ttk.Label(config_row, text="0.25")
        self.conf_display.pack(side='left', padx=(0, 20))
        
        # Atualizar display da confiança
        def update_conf_display(*args):
            self.conf_display.config(text=f"{self.confidence_threshold.get():.2f}")
        self.confidence_threshold.trace('w', update_conf_display)
        
        ttk.Label(config_row, text="Tamanho:").pack(side='left', padx=(20, 5))
        size_combo = ttk.Combobox(config_row, textvariable=self.img_size, 
                                 values=[320, 640, 1280], width=8, state='readonly')
        size_combo.pack(side='left')
        
        # Área principal
        main_content = ttk.Frame(main_frame)
        main_content.pack(fill='both', expand=True)
        
        # Frame da imagem
        image_frame = ttk.LabelFrame(main_content, text="🖼️ Análise da Imagem", padding=10)
        image_frame.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Canvas
        self.canvas = tk.Canvas(image_frame, bg='white', width=900, height=600)
        self.canvas.pack(fill='both', expand=True)
        
        # Informações da imagem
        self.image_info = ttk.Label(image_frame, text="Carregue uma imagem para começar", 
                                   font=('Arial', 10))
        self.image_info.pack(pady=5)
        
        # Frame de resultados
        results_frame = ttk.LabelFrame(
            main_content,
            text="🎯 Pontos Críticos Encontrados",
            padding=10,
            width=400   # largura em pixels
        )
        results_frame.pack(side='right', fill='y')  # width removido daqui

        # Se você quiser que o frame respeite sempre essa largura, pode desativar o pack-propagation:
        results_frame.pack_propagate(False)
        
        # Status da detecção
        self.detection_status = ttk.Label(results_frame, text="Aguardando detecção...", 
                                         font=('Arial', 11, 'bold'))
        self.detection_status.pack(pady=(0, 10))
        
        # Resumo
        summary_frame = ttk.LabelFrame(results_frame, text="📊 Resumo", padding=10)
        summary_frame.pack(fill='x', pady=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=6, font=('Arial', 9))
        self.summary_text.pack(fill='x')
        
        # Lista de detecções
        detections_frame = ttk.LabelFrame(results_frame, text="🔍 Detecções", padding=10)
        detections_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Treeview
        columns = ('ID', 'Tipo', 'Confiança', 'Posição')
        self.tree = ttk.Treeview(detections_frame, columns=columns, show='headings', height=12)
        
        self.tree.heading('ID', text='#')
        self.tree.heading('Tipo', text='Tipo')
        self.tree.heading('Confiança', text='Conf.')
        self.tree.heading('Posição', text='Posição')
        
        self.tree.column('ID', width=30)
        self.tree.column('Tipo', width=100)
        self.tree.column('Confiança', width=60)
        self.tree.column('Posição', width=80)
        
        tree_scroll = ttk.Scrollbar(detections_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        
        # Bind para seleção
        self.tree.bind('<<TreeviewSelect>>', self.on_detection_select)
        
        # Botões de ação
        action_frame = ttk.Frame(results_frame)
        action_frame.pack(fill='x')
        
        ttk.Button(action_frame, text="📋 Relatório HTML", 
                  command=self.generate_html_report).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="🖼️ Salvar Imagem", 
                  command=self.save_annotated_image).pack(fill='x', pady=2)
        
        self.update_summary()
        
    def load_model_automatically(self):
        """Carrega o modelo YOLO automaticamente"""
        if not HAS_YOLO:
            self.model_status.config(text="❌ YOLO não instalado! Execute: pip install ultralytics", 
                                   foreground='red')
            return
        
        # Procurar modelo
        search_paths = [
            "runs/detect/thermal_small2/weights/best.pt",
            "runs/detect/thermal_small/weights/best.pt", 
            "best.pt",
            "weights/best.pt",
            "models/best.pt"
        ]
        
        model_found = False
        
        for path in search_paths:
            if os.path.exists(path):
                try:
                    print(f"🔄 Tentando carregar modelo: {path}")
                    self.model = YOLO(path)
                    self.model_path = path
                    self.model_loaded = True
                    model_found = True
                    
                    self.model_status.config(
                        text=f"✅ Modelo carregado: {os.path.basename(path)}", 
                        foreground='green'
                    )
                    
                    # Testar modelo
                    print(f"📋 Classes do modelo: {self.model.names}")
                    print(f"🎯 Modelo pronto para detecção!")
                    
                    messagebox.showinfo("Modelo Carregado", 
                                      f"✅ Modelo YOLO carregado com sucesso!\n\n"
                                      f"📁 Arquivo: {path}\n"
                                      f"🎯 Classes: {len(self.model.names)}\n"
                                      f"🚀 Pronto para detectar pontos críticos!")
                    break
                    
                except Exception as e:
                    print(f"❌ Erro ao carregar {path}: {e}")
                    continue
        
        if not model_found:
            # Busca recursiva
            print("🔍 Procurando modelo recursivamente...")
            for root, dirs, files in os.walk("."):
                if "best.pt" in files:
                    path = os.path.join(root, "best.pt")
                    try:
                        print(f"🔄 Tentando carregar: {path}")
                        self.model = YOLO(path)
                        self.model_path = path
                        self.model_loaded = True
                        model_found = True
                        
                        self.model_status.config(
                            text=f"✅ Modelo encontrado: {os.path.basename(path)}", 
                            foreground='green'
                        )
                        
                        messagebox.showinfo("Modelo Encontrado", 
                                          f"✅ Modelo encontrado e carregado!\n\n{path}")
                        break
                        
                    except Exception as e:
                        print(f"❌ Erro: {e}")
                        continue
        
        if not model_found:
            self.model_status.config(
                text="❌ Modelo best.pt não encontrado! Coloque na pasta do programa", 
                foreground='red'
            )
            messagebox.showerror("Modelo Não Encontrado", 
                                "❌ Não foi possível encontrar o modelo best.pt\n\n"
                                "📋 Certifique-se de que o arquivo está em:\n"
                                "• runs/detect/thermal_small2/weights/best.pt\n"
                                "• ou na pasta raiz do programa\n\n"
                                "🔧 O modelo é necessário para detectar pontos críticos!")
    
    def load_new_image(self):
        """Carrega uma nova imagem para análise"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Nova Imagem Térmica",
            filetypes=[
                ("Imagens Térmicas", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos", "*.*")
            ]
        )
        
        if file_path:
            try:
                print(f"📁 Carregando imagem: {file_path}")
                
                # Carregar imagem
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise Exception("Não foi possível carregar a imagem")
                
                self.current_image = self.original_image.copy()
                self.current_image_path = file_path
                
                # Exibir imagem
                self.display_image(self.current_image)
                
                # Atualizar informações
                height, width = self.original_image.shape[:2]
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                
                self.image_info.config(
                    text=f"📁 {os.path.basename(file_path)} | "
                         f"📏 {width}x{height} | "
                         f"💾 {file_size:.1f}MB"
                )
                
                # Habilitar detecção
                if self.model_loaded:
                    self.detect_btn.config(state='normal')
                
                # Limpar resultados anteriores
                self.detection_results = []
                self.update_tree()
                self.update_summary()
                
                self.detection_status.config(text="✅ Imagem carregada - Pronta para análise")
                
                print(f"✅ Imagem carregada com sucesso!")
                
            except Exception as e:
                messagebox.showerror("Erro", f"❌ Erro ao carregar imagem:\n{str(e)}")
                print(f"❌ Erro ao carregar imagem: {e}")
    
    def detect_critical_points(self):
        """Detecta pontos críticos usando o modelo YOLO treinado"""
        if not self.model_loaded:
            messagebox.showwarning("Aviso", "❌ Modelo não carregado!")
            return
        
        if self.original_image is None:
            messagebox.showwarning("Aviso", "❌ Carregue uma imagem primeiro!")
            return
        
        def detection_thread():
            try:
                print(f"🔍 Iniciando detecção com modelo YOLO...")
                print(f"⚙️ Configurações: conf={self.confidence_threshold.get():.2f}, "
                      f"iou={self.iou_threshold.get():.2f}, size={self.img_size.get()}")
                
                # Atualizar status na UI
                self.root.after(0, lambda: self.detect_btn.config(text="🔄 DETECTANDO...", state='disabled'))
                self.root.after(0, lambda: self.detection_status.config(text="🔄 Analisando imagem com IA..."))
                
                # EXECUTAR DETECÇÃO REAL COM YOLO
                results = self.model(
                    self.current_image_path,
                    conf=self.confidence_threshold.get(),
                    iou=self.iou_threshold.get(), 
                    imgsz=self.img_size.get(),
                    verbose=True  # Para debug
                )
                
                print(f"📊 Resultados brutos do YOLO: {len(results)} resultado(s)")
                
                # Processar resultados REAIS
                detections = self.process_yolo_results(results[0])
                
                print(f"🎯 Detecções processadas: {len(detections)}")
                
                # Desenhar detecções na imagem
                annotated_image = self.draw_real_detections(self.original_image.copy(), detections)
                
                # Atualizar UI na thread principal
                self.root.after(0, lambda: self.detection_complete(detections, annotated_image))
                
            except Exception as e:
                error_msg = f"❌ Erro na detecção: {str(e)}"
                print(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Erro na Detecção", error_msg))
                self.root.after(0, lambda: self.detect_btn.config(text="🔍 DETECTAR PONTOS", state='normal'))
                self.root.after(0, lambda: self.detection_status.config(text="❌ Erro na detecção"))
        
        # Executar em thread separada
        threading.Thread(target=detection_thread, daemon=True).start()
    
    def process_yolo_results(self, results):
        """Processa os resultados REAIS do YOLO"""
        detections = []
        
        print(f"🔍 Processando resultados do YOLO...")
        
        if results.boxes is not None and len(results.boxes) > 0:
            # Extrair dados das detecções
            boxes = results.boxes.xyxy.cpu().numpy()  # Coordenadas x1,y1,x2,y2
            confidences = results.boxes.conf.cpu().numpy()  # Confiança
            classes = results.boxes.cls.cpu().numpy().astype(int)  # Classes
            
            print(f"📦 Boxes encontradas: {len(boxes)}")
            print(f"🎯 Classes detectadas: {set(classes)}")
            print(f"📊 Confianças: {confidences}")
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                # Verificar se a classe existe no nosso mapeamento
                if cls in self.class_mapping:
                    class_info = self.class_mapping[cls]
                    
                    detection = {
                        'id': i + 1,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2,
                        'area': (x2 - x1) * (y2 - y1),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': class_info['name'],
                        'display_name': class_info['display'],
                        'color': class_info['color'],
                        'severity': class_info['severity'],
                        'temp_estimate': class_info['temp_est'],
                        'priority': class_info['priority']
                    }
                    
                    detections.append(detection)
                    
                    print(f"✅ Detecção {i+1}: {class_info['display']} "
                          f"(conf: {conf:.3f}, pos: {x1},{y1}-{x2},{y2})")
                else:
                    print(f"⚠️ Classe desconhecida: {cls}")
        else:
            print("ℹ️ Nenhuma detecção encontrada pelo modelo")
        
        # Ordenar por prioridade (extremo primeiro) e depois por confiança
        detections.sort(key=lambda x: (x['priority'], -x['confidence']))
        
        print(f"🎯 Total de detecções válidas: {len(detections)}")
        return detections
    
    def draw_real_detections(self, image, detections):
        """Desenha as detecções REAIS na imagem"""
        print(f"🎨 Desenhando {len(detections)} detecções na imagem...")
        
        for detection in detections:
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            color = detection['color']
            severity = detection['severity']
            confidence = detection['confidence']
            display_name = detection['display_name']
            
            # Desenhar retângulo principal
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Preparar label
            label = f"{severity}: {confidence:.2f}"
            
            # Configurações do texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Calcular tamanho do texto
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Desenhar fundo do texto
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 10),
                (x1 + text_width + 10, y1),
                color, -1
            )
            
            # Cor do texto (contraste)
            text_color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
            
            # Desenhar texto
            cv2.putText(
                image, label,
                (x1 + 5, y1 - baseline - 5),
                font, font_scale, text_color, thickness
            )
            
            # Desenhar ID no centro
            id_text = f"#{detection['id']}"
            cv2.putText(
                image, id_text,
                (detection['center_x'] - 15, detection['center_y'] + 5),
                font, 0.8, text_color, 2
            )
            
            # Desenhar temperatura estimada
            temp_text = detection['temp_estimate']
            cv2.putText(
                image, temp_text,
                (x1 + 5, y2 + 20),
                font, 0.5, color, 1
            )
        
        print(f"✅ Detecções desenhadas com sucesso!")
        return image
    
    def detection_complete(self, detections, annotated_image):
        """Chamado quando a detecção é concluída"""
        print(f"🎉 Detecção concluída! {len(detections)} pontos encontrados")
        
        # Salvar resultados
        self.detection_results = detections
        self.current_image = annotated_image
        
        # Atualizar displays
        self.display_image(annotated_image)
        self.update_tree()
        self.update_summary()
        
        # Restaurar botão
        self.detect_btn.config(text="🔍 DETECTAR PONTOS", state='normal')
        
        # Atualizar status
        if len(detections) == 0:
            self.detection_status.config(text="✅ Análise concluída - Nenhum ponto crítico detectado")
            messagebox.showinfo("Análise Concluída", 
                              "✅ Análise concluída!\n\n"
                              "🎯 Nenhum ponto crítico detectado.\n"
                              "📊 Equipamento aparenta estar normal.\n\n"
                              "💡 Se esperava detecções, tente diminuir a sensibilidade.")
        else:
            # Contar por severidade
            by_severity = {}
            for det in detections:
                sev = det['severity']
                by_severity[sev] = by_severity.get(sev, 0) + 1
            
            self.detection_status.config(text=f"⚠️ {len(detections)} pontos críticos detectados!")
            
            # Mensagem detalhada
            message = f"⚠️ PONTOS CRÍTICOS DETECTADOS!\n\n"
            message += f"🎯 Total: {len(detections)} pontos\n\n"
            
            for severity in ['EXTREMO', 'CRÍTICO', 'ATENÇÃO']:
                count = by_severity.get(severity, 0)
                if count > 0:
                    if severity == 'EXTREMO':
                        message += f"🚨 {severity}: {count} (AÇÃO IMEDIATA!)\n"
                    elif severity == 'CRÍTICO':
                        message += f"🔴 {severity}: {count} (Manutenção necessária)\n"
                    else:
                        message += f"🟡 {severity}: {count} (Monitoramento)\n"
            
            message += f"\n📋 Verifique os detalhes na lista à direita."
            
            messagebox.showwarning("Pontos Críticos Detectados", message)
    
    def display_image(self, image):
        """Exibe imagem no canvas"""
        if image is None:
            return
        
        # Obter dimensões do canvas
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Redimensionar mantendo proporção
            height, width = image.shape[:2]
            scale = min(canvas_width/width, canvas_height/height, 1.0)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            resized = cv2.resize(image, (new_width, new_height))
            
            # Converter BGR para RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Exibir no canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=photo, anchor='center'
            )
            self.canvas.image = photo  # Manter referência
    
    def update_tree(self):
        """Atualiza a árvore de detecções"""
        # Limpar
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Adicionar detecções
        for detection in self.detection_results:
            # Tags para cores
            severity = detection['severity']
            tags = []
            if severity == 'EXTREMO':
                tags = ['extremo']
            elif severity == 'CRÍTICO':
                tags = ['critico']
            else:
                tags = ['atencao']
            
            self.tree.insert('', 'end', values=(
                f"#{detection['id']}",
                detection['display_name'],
                f"{detection['confidence']:.3f}",
                f"({detection['center_x']},{detection['center_y']})"
            ), tags=tags)
        
        # Configurar cores
        self.tree.tag_configure('extremo', background='#ffcdd2')  # Vermelho claro
        self.tree.tag_configure('critico', background='#ffe0b2')  # Laranja claro
        self.tree.tag_configure('atencao', background='#fff9c4')  # Amarelo claro
    
    def update_summary(self):
        """Atualiza resumo"""
        if not self.detection_results:
            summary = """
📊 RESUMO DA ANÁLISE

Status: Aguardando detecção
Pontos encontrados: 0

🔍 Para detectar pontos críticos:
1. Carregue uma imagem térmica
2. Clique em "DETECTAR PONTOS"
3. Aguarde a análise da IA

⚙️ Ajuste a sensibilidade se necessário
            """
        else:
            total = len(self.detection_results)
            
            # Contar por severidade
            by_severity = {}
            for det in self.detection_results:
                sev = det['severity']
                by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # Determinar status geral
            if by_severity.get('EXTREMO', 0) > 0:
                status = "🚨 SITUAÇÃO CRÍTICA"
            elif by_severity.get('CRÍTICO', 0) > 0:
                status = "⚠️ ATENÇÃO NECESSÁRIA"
            elif by_severity.get('ATENÇÃO', 0) > 0:
                status = "🔍 MONITORAMENTO"
            else:
                status = "✅ NORMAL"
            
            summary = f"""
📊 RESUMO DA ANÁLISE

Status: {status}
Pontos encontrados: {total}

🎯 Por severidade:
🚨 Extremos: {by_severity.get('EXTREMO', 0)}
🔴 Críticos: {by_severity.get('CRÍTICO', 0)}
🟡 Atenção: {by_severity.get('ATENÇÃO', 0)}

📁 Arquivo: {os.path.basename(self.current_image_path) if self.current_image_path else 'N/A'}
            """
        
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', summary)
    
    def on_detection_select(self, event):
        """Quando uma detecção é selecionada"""
        selection = self.tree.selection()
        if selection and self.detection_results:
            item = self.tree.item(selection[0])
            detection_id = int(item['values'][0].replace('#', ''))
            
            # Encontrar detecção
            detection = next((d for d in self.detection_results if d['id'] == detection_id), None)
            
            if detection:
                details = f"""
DETECÇÃO #{detection['id']} - {detection['display_name']}

🔥 Severidade: {detection['severity']}
🌡️ Temperatura estimada: {detection['temp_estimate']}
📊 Confiança: {detection['confidence']:.3f}

📍 Posição central: ({detection['center_x']}, {detection['center_y']})
📏 Dimensões: {detection['width']} × {detection['height']} pixels
📐 Área: {detection['area']} px²

💡 Recomendação:
{self.get_recommendation(detection['severity'])}
                """
                
                messagebox.showinfo(f"Detalhes - {detection['display_name']}", details)
    
    def get_recommendation(self, severity):
        """Recomendações por severidade"""
        recommendations = {
            'EXTREMO': "🚨 AÇÃO IMEDIATA! Verificar equipamento urgentemente. Risco de falha iminente ou dano.",
            'CRÍTICO': "⚠️ Atenção necessária. Agendar manutenção prioritária dentro de 7 dias.",
            'ATENÇÃO': "🔍 Monitoramento recomendado. Incluir na próxima manutenção programada."
        }
        return recommendations.get(severity, "Verificar equipamento conforme procedimentos.")
    
    def batch_analyze(self):
        """Análise em lote de uma pasta"""
        if not self.model_loaded:
            messagebox.showwarning("Aviso", "❌ Modelo não carregado!")
            return
        
        folder = filedialog.askdirectory(title="Selecionar Pasta com Imagens Térmicas")
        if not folder:
            return
        
        # Encontrar imagens
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
            image_files.extend(glob.glob(os.path.join(folder, ext.upper())))
        
        if not image_files:
            messagebox.showwarning("Aviso", "❌ Nenhuma imagem encontrada na pasta!")
            return
        
        result = messagebox.askyesno("Análise em Lote", 
                                   f"🔍 Encontradas {len(image_files)} imagens.\n\n"
                                   f"Analisar todas automaticamente?")
        if result:
            self.run_batch_analysis(image_files)
    
    def run_batch_analysis(self, image_files):
        """Executa análise em lote"""
        def batch_thread():
            batch_results = []
            
            for i, img_path in enumerate(image_files):
                try:
                    # Atualizar status
                    self.root.after(0, lambda i=i, total=len(image_files): 
                                   self.detection_status.config(text=f"🔄 Analisando {i+1}/{total}..."))
                    
                    # Detectar com YOLO
                    results = self.model(img_path, conf=self.confidence_threshold.get(), verbose=False)
                    detections = self.process_yolo_results(results[0])
                    
                    # Salvar resultado
                    batch_results.append({
                        'file': os.path.basename(img_path),
                        'path': img_path,
                        'detections': len(detections),
                        'details': detections
                    })
                    
                except Exception as e:
                    print(f"❌ Erro em {img_path}: {e}")
            
            # Finalizar
            self.root.after(0, lambda: self.batch_complete(batch_results))
        
        threading.Thread(target=batch_thread, daemon=True).start()
    
    def batch_complete(self, results):
        """Análise em lote concluída"""
        total_images = len(results)
        total_detections = sum(r['detections'] for r in results)
        problematic = len([r for r in results if r['detections'] > 0])
        
        # Mostrar resumo
        summary = f"""
🎉 ANÁLISE EM LOTE CONCLUÍDA!

📊 Resumo:
• Imagens analisadas: {total_images}
• Total de pontos críticos: {total_detections}
• Imagens com problemas: {problematic}
• Taxa de problemas: {(problematic/total_images)*100:.1f}%

🔥 Imagens mais críticas:
"""
        
        # Top 5 mais problemáticas
        sorted_results = sorted(results, key=lambda x: x['detections'], reverse=True)
        for result in sorted_results[:5]:
            if result['detections'] > 0:
                summary += f"• {result['file']}: {result['detections']} pontos\n"
        
        self.detection_status.config(text="✅ Análise em lote concluída!")
        messagebox.showinfo("Análise em Lote Concluída", summary)
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"relatorio_lote_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total_images': total_images,
                        'total_detections': total_detections,
                        'problematic_images': problematic
                    },
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Relatório Salvo", f"📋 Relatório salvo em: {report_file}")
        except Exception as e:
            print(f"❌ Erro ao salvar relatório: {e}")
    
    def save_results(self):
        """Salva resultados da análise atual"""
        if not self.detection_results:
            messagebox.showwarning("Aviso", "❌ Nenhum resultado para salvar!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Resultados",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        
        if file_path:
            try:
                data = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'image_file': os.path.basename(self.current_image_path) if self.current_image_path else 'unknown',
                        'model_file': os.path.basename(self.model_path) if self.model_path else 'unknown',
                        'total_detections': len(self.detection_results)
                    },
                    'settings': {
                        'confidence_threshold': self.confidence_threshold.get(),
                        'iou_threshold': self.iou_threshold.get(),
                        'image_size': self.img_size.get()
                    },
                    'detections': self.detection_results
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Sucesso", f"✅ Resultados salvos em:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Erro", f"❌ Erro ao salvar:\n{str(e)}")
    
    def generate_html_report(self):
        """Gera relatório HTML"""
        if not self.detection_results:
            messagebox.showwarning("Aviso", "❌ Nenhum resultado para relatório!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Relatório HTML",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("Todos", "*.*")]
        )
        
        if file_path:
            try:
                html_content = self.create_html_report()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                messagebox.showinfo("Sucesso", f"✅ Relatório gerado:\n{file_path}")
                
                # Abrir no navegador
                try:
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(file_path)}")
                except:
                    pass
                
            except Exception as e:
                messagebox.showerror("Erro", f"❌ Erro ao gerar relatório:\n{str(e)}")
    
    def create_html_report(self):
        """Cria relatório HTML detalhado"""
        # Contar por severidade
        by_severity = {}
        for det in self.detection_results:
            sev = det['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Detecção Térmica - IA REAL</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #333; border-bottom: 3px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
        .model-info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .alert {{ padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .alert-danger {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
        .alert-warning {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
        .alert-info {{ background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .extremo {{ background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); }}
        .critico {{ background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); }}
        .atencao {{ background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .detection-extremo {{ background-color: #ffebee; }}
        .detection-critico {{ background-color: #fff3e0; }}
        .detection-atencao {{ background-color: #e3f2fd; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Relatório de Detecção Térmica</h1>
            <h2>Análise com IA YOLO Treinada</h2>
            <p><strong>Data:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </div>
        
        <div class="model-info">
            <h3>🤖 Informações do Modelo</h3>
            <p><strong>Modelo:</strong> {os.path.basename(self.model_path) if self.model_path else 'N/A'}</p>
            <p><strong>Imagem:</strong> {os.path.basename(self.current_image_path) if self.current_image_path else 'N/A'}</p>
            <p><strong>Configurações:</strong> Confiança: {self.confidence_threshold.get():.2f}, Tamanho: {self.img_size.get()}px</p>
        </div>
        """
        
        # Alerta baseado na severidade
        if by_severity.get('EXTREMO', 0) > 0:
            html += f"""
        <div class="alert alert-danger">
            <h3>🚨 SITUAÇÃO CRÍTICA DETECTADA</h3>
            <p><strong>{by_severity['EXTREMO']} pontos extremos</strong> detectados pela IA. Ação imediata necessária!</p>
        </div>
            """
        elif by_severity.get('CRÍTICO', 0) > 0:
            html += f"""
        <div class="alert alert-warning">
            <h3>⚠️ ATENÇÃO NECESSÁRIA</h3>
            <p><strong>{by_severity['CRÍTICO']} pontos críticos</strong> detectados. Manutenção prioritária recomendada.</p>
        </div>
            """
        else:
            html += f"""
        <div class="alert alert-info">
            <h3>ℹ️ MONITORAMENTO</h3>
            <p>Apenas zonas de atenção detectadas. Monitoramento contínuo recomendado.</p>
        </div>
            """
        
        html += f"""
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(self.detection_results)}</div>
                <div class="stat-label">Total Detectado</div>
            </div>
            <div class="stat-card extremo">
                <div class="stat-value">{by_severity.get('EXTREMO', 0)}</div>
                <div class="stat-label">Pontos Extremos</div>
            </div>
            <div class="stat-card critico">
                <div class="stat-value">{by_severity.get('CRÍTICO', 0)}</div>
                <div class="stat-label">Pontos Críticos</div>
            </div>
            <div class="stat-card atencao">
                <div class="stat-value">{by_severity.get('ATENÇÃO', 0)}</div>
                <div class="stat-label">Pontos de Atenção</div>
            </div>
        </div>
        
        <h2>🎯 Detecções da IA</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Tipo</th>
                    <th>Severidade</th>
                    <th>Confiança IA</th>
                    <th>Posição</th>
                    <th>Área</th>
                    <th>Temp. Estimada</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for detection in self.detection_results:
            severity = detection['severity']
            row_class = f"detection-{severity.lower()}"
            
            html += f"""
                <tr class="{row_class}">
                    <td>#{detection['id']}</td>
                    <td>{detection['display_name']}</td>
                    <td>{severity}</td>
                    <td>{detection['confidence']:.3f}</td>
                    <td>({detection['center_x']}, {detection['center_y']})</td>
                    <td>{detection['area']} px²</td>
                    <td>{detection['temp_estimate']}</td>
                </tr>
            """
        
        html += f"""
            </tbody>
        </table>
        
        <h2>🛠️ Recomendações Baseadas na IA</h2>
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px;">
        """
        
        if by_severity.get('EXTREMO', 0) > 0:
            html += f"""
            <h3>🚨 Ações Imediatas (Pontos Extremos):</h3>
            <ul>
                <li>Verificar equipamento IMEDIATAMENTE</li>
                <li>Considerar desligamento de emergência</li>
                <li>Chamar equipe especializada</li>
                <li>Documentar todas as ações</li>
            </ul>
            """
        
        if by_severity.get('CRÍTICO', 0) > 0:
            html += f"""
            <h3>⚠️ Ações Prioritárias (Pontos Críticos):</h3>
            <ul>
                <li>Agendar manutenção em até 7 dias</li>
                <li>Verificar conexões e isolamentos</li>
                <li>Monitorar temperatura diariamente</li>
                <li>Preparar plano de contingência</li>
            </ul>
            """
        
        html += f"""
        </div>
        
        <div class="footer">
            <p><strong>Detector Térmico com IA YOLO v2.0</strong></p>
            <p>Relatório gerado automaticamente usando modelo treinado - Consulte especialista para decisões críticas</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def save_annotated_image(self):
        """Salva imagem com anotações"""
        if self.current_image is None:
            messagebox.showwarning("Aviso", "❌ Nenhuma imagem para salvar!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Imagem Anotada",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_image)
                messagebox.showinfo("Sucesso", f"✅ Imagem salva em:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"❌ Erro ao salvar:\n{str(e)}")

def main():
    """Função principal"""
    print("🚀 Iniciando Detector Térmico REAL...")
    
    root = tk.Tk()
    app = RealThermalDetector(root)
    
    # Centralizar janela
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    print("✅ Interface carregada!")
    root.mainloop()

if __name__ == "__main__":
    main()