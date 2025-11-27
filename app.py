import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QComboBox, QCheckBox, QSpinBox, 
                             QDoubleSpinBox, QGroupBox, QScrollArea, QTextEdit,
                             QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from Controller.SolverBackend import SolverBackend

class MatrixSolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Solver Pro")
        self.setGeometry(100, 100, 1100, 800)
        
        # --- Main Layout Setup ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20) # Add breathing room
        self.main_layout.setSpacing(15)

        # --- Header ---
        header_label = QLabel("Linear Algebra Solver")
        header_label.setObjectName("HeaderLabel") # For styling
        header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(header_label)

        # --- 1. Configuration Card ---
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout()
        config_layout.setContentsMargins(15, 20, 15, 15)

        # Dimension
        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Dimension (N):"))
        self.n_spinner = QSpinBox()
        self.n_spinner.setRange(2, 10)
        self.n_spinner.setValue(3)
        self.n_spinner.setFixedWidth(60)
        self.n_spinner.valueChanged.connect(self.generate_matrix_grid)
        dim_layout.addWidget(self.n_spinner)
        config_layout.addLayout(dim_layout)
        
        config_layout.addStretch() # Spacer

        # Method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Solving Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Gaussian Elimination", 
            "LU Decomposition", 
            "Cholesky Decomposition", 
            "Gaussian Jordan", 
            "Jacobi Iteration", 
            "Gauss-Seidel Iteration"
        ])
        self.method_combo.setFixedWidth(200)
        self.method_combo.currentTextChanged.connect(self.toggle_iterative_params)
        method_layout.addWidget(self.method_combo)
        config_layout.addLayout(method_layout)

        config_group.setLayout(config_layout)
        self.main_layout.addWidget(config_group)

        # --- 2. Iterative Parameters (Hidden Card) ---
        self.iter_group = QGroupBox("Iterative Parameters")
        iter_layout = QHBoxLayout()
        iter_layout.setContentsMargins(15, 20, 15, 15)
        
        iter_layout.addWidget(QLabel("Sig Figures:"))
        self.sig_figs_input = QSpinBox()
        self.sig_figs_input.setRange(1, 15) 
        self.sig_figs_input.setValue(4)     
        iter_layout.addWidget(self.sig_figs_input)

        iter_layout.addWidget(QLabel("Tolerance:"))
        self.tol_input = QDoubleSpinBox()
        self.tol_input.setDecimals(9)       
        self.tol_input.setValue(0.0001)
        self.tol_input.setSingleStep(0.00001)
        iter_layout.addWidget(self.tol_input)

        iter_layout.addWidget(QLabel("Max Iterations:"))
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 1000)
        self.max_iter_input.setValue(50)
        iter_layout.addWidget(self.max_iter_input)

        self.iter_group.setLayout(iter_layout)
        self.main_layout.addWidget(self.iter_group)
        self.iter_group.hide() 

        # --- 3. Matrix Input Area (Card style) ---
        input_container = QGroupBox("Matrix Input")
        input_layout = QVBoxLayout()
        input_container.setLayout(input_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame) # Remove ugly scroll border
        
        self.matrix_container = QWidget()
        self.matrix_container.setObjectName("MatrixContainer") # For styling white bg
        self.matrix_layout = QGridLayout(self.matrix_container)
        self.matrix_layout.setSpacing(10)
        
        self.scroll_area.setWidget(self.matrix_container)
        input_layout.addWidget(self.scroll_area)
        
        self.main_layout.addWidget(input_container)

        self.matrix_inputs = [] 
        self.vector_b_inputs = []

        self.generate_matrix_grid()

        # --- 4. Controls ---
        controls_layout = QHBoxLayout()
        
        self.chk_steps = QCheckBox("Show Detailed Steps")
        self.chk_steps.setChecked(True)
        self.chk_steps.setStyleSheet("font-size: 14px; color: #555;")
        controls_layout.addWidget(self.chk_steps)

        controls_layout.addStretch()

        self.btn_solve = QPushButton("SOLVE SYSTEM")
        self.btn_solve.setObjectName("SolveButton") # ID for custom styling
        self.btn_solve.setCursor(Qt.PointingHandCursor)
        self.btn_solve.clicked.connect(self.solve_system)
        controls_layout.addWidget(self.btn_solve)
        
        self.main_layout.addLayout(controls_layout)

        # --- 5. Output Area ---
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setObjectName("ConsoleOutput")
        self.output_display.setPlaceholderText("Computation results will appear here...")
        self.main_layout.addWidget(self.output_display)

        # Initialize Backend    
        self.backend = SolverBackend()
        

    def generate_matrix_grid(self):
        # Clear existing
        for i in reversed(range(self.matrix_layout.count())): 
            self.matrix_layout.itemAt(i).widget().setParent(None)
        
        self.matrix_inputs = []
        self.vector_b_inputs = []
        n = self.n_spinner.value()

        # Headers
        self.matrix_layout.addWidget(QLabel("Matrix A (Coefficients)"), 0, 0, 1, n, Qt.AlignCenter)
        self.matrix_layout.addWidget(QLabel("Vector b"), 0, n + 1, 1, 1, Qt.AlignCenter)
        
        # Grid
        for row in range(n):
            row_inputs = []
            for col in range(n):
                inp = QLineEdit()
                inp.setPlaceholderText("0")
                inp.setAlignment(Qt.AlignCenter)
                self.matrix_layout.addWidget(inp, row + 1, col)
                row_inputs.append(inp)
            self.matrix_inputs.append(row_inputs)
            
            # Equality Sign
            eq_label = QLabel("=")
            eq_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #7f8c8d;")
            self.matrix_layout.addWidget(eq_label, row + 1, n)
            
            # Vector B
            b_inp = QLineEdit()
            b_inp.setPlaceholderText("0")
            b_inp.setAlignment(Qt.AlignCenter)
            b_inp.setObjectName("VectorBInput") # Style differently
            self.matrix_layout.addWidget(b_inp, row + 1, n + 1)
            self.vector_b_inputs.append(b_inp)

    def toggle_iterative_params(self, method_name):
        if "Iteration" in method_name:
            self.iter_group.show()
        else:
            self.iter_group.hide()

    def get_matrix_data(self):
        n = self.n_spinner.value()
        A = []
        b = []
        try:
            for row in range(n):
                current_row = []
                for col in range(n):
                    text = self.matrix_inputs[row][col].text()
                    if not text: text = "0" # Default to 0 if empty
                    current_row.append(float(text))
                A.append(current_row)
            
            for row in range(n):
                text = self.vector_b_inputs[row].text()
                if not text: text = "0"
                b.append(float(text))
            return A, b
        except ValueError as e:
            self.output_display.setText(f"Input Error: Please enter valid numbers.\nDetails: {str(e)}")
            return None, None

    def solve_system(self):
        A, b = self.get_matrix_data()
        if A is None: return 
        
        method = self.method_combo.currentText()
        tol = self.tol_input.value()
        max_iter = self.max_iter_input.value()
        sig_figs = self.sig_figs_input.value()

        self.output_display.clear()
        
        try:
            # Dispatcher Call
            result = self.backend.solve(method, A, b, tol=tol, max_iter=max_iter)
            
            # --- Result Handling ---
            if method == "Cholesky Decomposition":
                x, L, steps = result
                
                if self.chk_steps.isChecked():
                    self.output_display.append("<h3>--- Step-by-Step Trace ---</h3>")
                    for step in steps:
                        # Simple HTML formatting for better readability
                        if "Step" in step:
                            self.output_display.append(f"<b>{step}</b>")
                        else:
                            self.output_display.append(step)
                    self.output_display.append("<hr>")
                
                result_text = "<h3>--- Final Results ---</h3>"
                result_text += "<b>Solution Vector (x):</b><br>" + str([round(val, 4) for val in x])
                self.output_display.append(result_text)
            
            else:
                self.output_display.append(f"<b>Method '{method}' is not implemented yet!</b>")

        except Exception as e:
            self.output_display.setText(f"Calculation Error:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # --- MODERN STYLESHEET ---
    style_sheet = """
        QMainWindow {
            background-color: #f4f7f6;
        }
        
        QLabel {
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            color: #34495e;
        }
        
        QLabel#HeaderLabel {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }

        QGroupBox {
            background-color: white;
            border-radius: 8px;
            border: 1px solid #dcdcdc;
            margin-top: 20px; /* Space for title */
            font-weight: bold;
            color: #2c3e50;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 10px;
            left: 10px;
            color: #2c3e50;
        }

        QLineEdit {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 8px;
            background-color: #ffffff;
            font-size: 14px;
            selection-background-color: #3498db;
        }
        
        QLineEdit:focus {
            border: 2px solid #3498db;
            background-color: #ecf0f1;
        }
        
        QLineEdit#VectorBInput {
            background-color: #fcf3cf; /* Light yellow for Vector B */
            border: 1px solid #f1c40f;
        }

        QComboBox, QSpinBox, QDoubleSpinBox {
            padding: 6px;
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            background-color: white;
            font-size: 13px;
        }

        QPushButton#SolveButton {
            background-color: #27ae60;
            color: white;
            border-radius: 5px;
            padding: 12px 25px;
            font-size: 15px;
            font-weight: bold;
            border: none;
        }
        
        QPushButton#SolveButton:hover {
            background-color: #2ecc71;
        }
        
        QPushButton#SolveButton:pressed {
            background-color: #219150;
        }

        QTextEdit#ConsoleOutput {
            background-color: #2c3e50;
            color: #ecf0f1;
            border: 1px solid #34495e;
            border-radius: 6px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
            padding: 10px;
        }
        
        QWidget#MatrixContainer {
            background-color: white;
        }
    """
    app.setStyleSheet(style_sheet)
    
    window = MatrixSolverGUI()
    window.show()
    sys.exit(app.exec_())