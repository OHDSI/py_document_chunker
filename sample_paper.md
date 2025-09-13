# Abstract
This paper introduces a novel method for quantum key distribution. We demonstrate a new protocol that is resistant to common eavesdropping attacks.

# 1. Introduction
Quantum Key Distribution (QKD) is a secure communication method that implements a cryptographic protocol involving components of quantum mechanics. It enables two parties to produce a shared random secret key known only to them, which can then be used to encrypt and decrypt messages.

## 1.1 Prior Work
Existing QKD protocols, such as BB84, are theoretically secure. However, their implementation in real-world devices can be vulnerable to side-channel attacks. Our work builds upon the foundations laid by these pioneering protocols.

# 2. Methods
Our proposed protocol, which we call the "Resistant QKD Protocol" (RQP), uses a three-state system instead of the traditional two-state system.

The core of the RQP is the following algorithm:
```python
def resistant_qkd_protocol(alice_basis, bob_basis):
    # Simulate the quantum channel
    alice_qubits = prepare_qubits(alice_basis)
    # Eavesdropper interaction would be modeled here
    measured_qubits = measure_qubits(alice_qubits, bob_basis)
    return sift_keys(measured_qubits)
```

## 2.1 Experimental Setup
The experiment was conducted using a fiber-optic cable network spanning 50 kilometers. A single-photon detector with a 99.5% efficiency was used.

### 2.1.1 Error Correction
We applied a Low-Density Parity-Check (LDPC) code for error correction, which proved to be highly efficient.

# 3. Results
The RQP achieved a secure key rate of 1.2 kbps over the 50km distance, which is a significant improvement over existing methods under similar conditions.

The results are summarized in the table below:

| Protocol | Distance (km) | Key Rate (kbps) |
|----------|---------------|-----------------|
| BB84     | 50            | 0.4             |
| RQP      | 50            | 1.2             |

# 4. Conclusion
The Resistant QKD Protocol offers a more secure and efficient method for quantum key distribution. Future work will involve testing the protocol over longer distances and with different types of optical fiber.
