import numpy as np

def print_case(case, description):
  print(f"\n👉 \033[1mCase: {case}\033[0m")
  print(f"   📝 {description}")
  
def evaluate_result(case, expected_r, expected_z, expected_status, r, z, status):
  print(f"\n👉 \033[1mCase: {case}\033[0m")
  print(f"   📝 Resultado esperado: r = {expected_r}, z = {expected_z}, status = {expected_status}")
  print(f"   🧪 Resultado obtenido: r = {r}, z = {z}, status = {status}")
  assert np.allclose(r, expected_r), f"r: {r} != {expected_r}"
  assert np.allclose(z, expected_z), f"z: {z} != {expected_z}"
  assert status == expected_status, f"status: {status} != {expected_status}"
  print("   ✅ Test passed")