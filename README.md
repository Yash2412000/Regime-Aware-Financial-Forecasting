# Regime-Aware Explainable Multi-Modal Financial Forecasting

## Overview

This repository contains the implementation of a **regime-aware, explainable
multi-modal machine learning framework for financial forecasting**.

The project explores whether combining multiple forecasting experts with market
regime information can improve adaptability in financial time-series prediction.
The proposed framework uses a **Mixture of Experts (MoE)** architecture where
different forecasting models contribute dynamically based on the detected market
conditions.

---

## Framework Architecture

The proposed framework consists of four main components:

1. **Multi-Modal Input Representation**
   - Historical market data
   - Technical indicators
   - Financial sentiment information
   - Macroeconomic context

2. **Market Regime Detection**
   - Hidden Markov Model (HMM)
   - Identification of different market conditions:
     - Bull
     - Bear
     - High volatility

3. **Forecasting Experts**
   
   Multiple forecasting models are used as specialised experts:

   - Long Short-Term Memory (LSTM)
   - Gated Recurrent Unit (GRU)
   - Transformer
   - Linear forecasting model

4. **Adaptive Gating Network**

   The gating network dynamically assigns weights to each expert depending on
   the identified market regime.

---

## Repository Structure
