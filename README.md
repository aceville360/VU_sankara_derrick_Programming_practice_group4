# VU_sankara_derrick_programming_group4


**Topic:** Object-Oriented Programming in Python

**Group Members:**
- Sankara Derrick (VU-BAD-2603-0551-DAY)
- Luyirika Isaac (VU-BAD-2607-4820-DAY)
- Nanyondo Zaituni (VU-BAD-2603-1526-DAY)
- Mazima Edmond (VU-BAD-2607-0445-DAY)
- Kiggundu Pius (VU-BAD-2607-0983-EVE)

## Project Overview
This repository contains a Farm Management System built with Python. The program demonstrates core Object-Oriented Programming (OOP) principles by reading a Kaggle dataset (`poultry.csv`) and converting each data row into a secure, self-contained object. 

## OOP Concepts Demonstrated

### 1. Encapsulation (Primary Focus)
The `PoultryBatch` class restricts direct access to its internal state to prevent invalid data corruption.
- **Private Attributes:** The `__feed_inventory_kg` and `__feed_per_bird_grams` (configured to a realistic 11-gram daily requirement for layer chickens) are protected using double underscores to invoke Python's name mangling.
- **Setters (Gatekeeping):** The `add_feed()` method acts as a gatekeeper. If external code attempts to pass a negative feed value, the setter intercepts and rejects it, guaranteeing the object's mathematical state remains valid.
- **Getters:** The `get_inventory()` method allows the main program to safely view the hidden data without exposing the variable to unauthorized modification.

### 2. Modules and Packages
The system avoids single-file clutter by utilizing modular architecture:
- `livestock.py`: A dedicated module containing the class blueprint.
- `main.py`: The execution script that handles data ingestion and object instantiation.

## Included Files
- `main.py`: The entry point that executes the logic and tests the encapsulation rules.
- `livestock.py`: The module containing the encapsulated `PoultryBatch` class.
- `poultry.csv`: The dataset containing batch IDs, dates, active bird counts, and mortality rates.
- `GROUP 4 OOP 3.pptx`: The group presentation slides detailing OOP theory.
- `Presentation_Report`: The comprehensive written report supporting the code and slides.

## How to Run the Code
1. Ensure `main.py`, `livestock.py`, and `poultry.csv` are located in the exact same directory.
2. Open your terminal in that directory.
3. Execute the main script:
   ```bash
   python main.py