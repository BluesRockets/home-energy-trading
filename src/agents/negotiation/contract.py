import json
import os
from web3 import Web3

# Replace with your Ethereum node provider URL (e.g., Infura, Alchemy) for Base Testnet
provider_url = "https://sepolia.base.org"
w3 = Web3(Web3.HTTPProvider(provider_url))

if not w3.is_connected():
    raise ConnectionError("Failed to connect to the Base node.")

# Set up your account details
private_key = "0x79713ff51b23a792fcd08650657dd278a0b5d5e05fba2ac3d676481779267f5a"  # NEVER expose your private key in production code
account = w3.eth.account.from_key(private_key)
print("Using account:", account.address)

# Replace with your deployed Uniswap V contract address
contract_address = Web3.to_checksum_address("0x489ef234b41b12ba2bb9033988ae707f7e951e93")

# Load the contract ABI from a JSON file located in the same directory as this script
abi_path = os.path.join(os.path.dirname(__file__), 'UniswapV1ABI.json')
with open(abi_path, 'r') as abi_file:
    contract_abi = json.load(abi_file)

# Create the Uniswap V contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Set chain parameters
CHAIN_ID = 84532 
GAS_LIMIT = 200000
GAS_PRICE = w3.to_wei('5', 'gwei')

# Define a minimal ERC20 ABI for allowance and approve
erc20_abi = [
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]

# Replace with the actual token contract addresses for Dollar and Electricity
# Update these addresses accordingly

dollar_token_address = Web3.to_checksum_address("0xcf90bE3E42359918ec74fcEb4aFb02A9E6C6A5cB")
electricity_token_address = Web3.to_checksum_address("0xC13aBB23A1258F2D1408D78912B475c6B560aFef")

# Create contract instances for the tokens

dollar_token = w3.eth.contract(address=dollar_token_address, abi=erc20_abi)
electricity_token = w3.eth.contract(address=electricity_token_address, abi=erc20_abi)


def build_and_send_tx(tx):
    """Helper function to sign and send a transaction, then wait for receipt."""
    nonce = w3.eth.get_transaction_count(account.address)
    tx.update({
        'chainId': CHAIN_ID,
        'gas': GAS_LIMIT,
        # 'gasPrice': GAS_PRICE,
        'nonce': nonce,
        'from': account.address,
    })
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print("Transaction sent. Tx hash:", tx_hash.hex())
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Transaction receipt:", receipt)
    return receipt


def check_and_approve(token_contract, token_name, required_amount):
    """Checks if the allowance for the given token is sufficient, and if not, approves the contract to spend the tokens.
    :param token_contract: The token contract instance
    :param token_name: A string representing the token name (for logging purposes)
    :param required_amount: The amount required (in smallest unit)
    """
    current_allowance = token_contract.functions.allowance(account.address, contract_address).call()
    if current_allowance < required_amount:
        print(f"{token_name} allowance is insufficient (current: {current_allowance}). Approving {required_amount}...")
        # Consider approving a much higher amount to avoid repeated approvals
        # For example, use max uint256:
        max_uint256 = 2**256 - 1
        approve_tx = token_contract.functions.approve(contract_address, max_uint256).build_transaction({'from': account.address})
        build_and_send_tx(approve_tx)
        # Re-read allowance after approval
        current_allowance = token_contract.functions.allowance(account.address, contract_address).call()
        print(f"New {token_name} allowance: {current_allowance}")
        if current_allowance < required_amount:
            raise Exception(f"Approval failed: {token_name} allowance is still insufficient.")
    else:
        print(f"{token_name} allowance is sufficient: {current_allowance}")


def add_liquidity_dollar_electricity(dollar_amount, electricity_amount):
    """Adds liquidity to the Uniswap V contract using Dollar and Electricity tokens.
    :param dollar_amount: The amount of Dollar tokens to deposit (in smallest unit)
    :param electricity_amount: The amount of Electricity tokens to deposit (in smallest unit)
    :return: Transaction receipt
    """
    # Check and approve Dollar and Electricity tokens if necessary
    check_and_approve(dollar_token, "Dollar", dollar_amount)
    check_and_approve(electricity_token, "Electricity", electricity_amount)
    tx = contract.functions.addLiquidity(dollar_amount, electricity_amount).build_transaction({})
    return build_and_send_tx(tx)


def remove_liquidity(liquidity_amount):
    """Removes liquidity from the Uniswap V contract.
    :param liquidity_amount: The amount of liquidity tokens to burn
    :return: Transaction receipt
    """
    tx = contract.functions.removeLiquidity(liquidity_amount).build_transaction({})
    return build_and_send_tx(tx)


def swap_dollar_to_electricity(dollar_in, min_electricity_out):
    """Swaps Dollar tokens for Electricity tokens using the contract.
    :param dollar_in: The amount of Dollar tokens to swap (in smallest unit)
    :param min_electricity_out: The minimum amount of Electricity tokens to receive (in smallest unit) to protect against slippage
    :return: Transaction receipt
    """
    # Check and approve Dollar token if necessary
    check_and_approve(dollar_token, "Dollar", dollar_in)
    tx = contract.functions.tokenAToTokenBSwap(dollar_in, min_electricity_out).build_transaction({'from': account.address})
    print(tx)
    return build_and_send_tx(tx)


def swap_electricity_to_dollar(electricity_in, min_dollar_out):
    """Swaps Electricity tokens for Dollar tokens using the contract.
    :param electricity_in: The amount of Electricity tokens to swap (in smallest unit)
    :param min_dollar_out: The minimum amount of Dollar tokens to receive (in smallest unit) to protect against slippage
    :return: Transaction receipt
    """
    # Check and approve Electricity token if necessary
    check_and_approve(electricity_token, "Electricity", electricity_in)
    tx = contract.functions.tokenBToTokenASwap(electricity_in, min_dollar_out).build_transaction({'from': account.address})
    return build_and_send_tx(tx)


if __name__ == "__main__":
    # Example usage:
    # Swap 1 unit of Dollar (adjust decimals accordingly) for Electricity
    dollar_in = w3.to_wei(1, 'ether')
    min_electricity_out = 1           # Minimal acceptable Electricity token output to protect against slippage
    print("Executing swap_dollar_to_electricity...")
    swap_dollar_to_electricity(dollar_in, min_electricity_out)

    # Uncomment below to test other functions
    #
    # Example: Add liquidity
    # dollar_amount = w3.toWei(1, 'ether')
    # electricity_amount = w3.toWei(2, 'ether')  # Adjust according to token ratio
    # print("Adding liquidity for Dollar and Electricity tokens...")
    # add_liquidity_dollar_electricity(dollar_amount, electricity_amount)
    #
    # Example: Remove liquidity
    # liquidity_amount = 100  # Replace with the actual liquidity token amount you want to remove
    # print("Removing liquidity...")
    # remove_liquidity(liquidity_amount)
    #
    # Example: Swap Electricity for Dollar
    # electricity_in = w3.toWei(1, 'ether')
    # min_dollar_out = 1
    # print("Executing swap_electricity_to_dollar...")
    # swap_electricity_to_dollar(electricity_in, min_dollar_out)