# Cartrack Database Tools Summary

This document provides a comprehensive overview of all database tools available in CartrackSQLDatabase.py for building call center AI agents. Each tool is annotated with `@tool` decorator for LangChain integration.

## **Core Client Information Tools**

### `get_client_profile(user_id: str) -> Optional[Dict]`
**Purpose**: Retrieves comprehensive client profile data including personal information, contact details, addresses, vehicles, and account information.

**Use Cases**: 
- Look up complete client information
- Verify contact details, address, or vehicle information
- Create reports or summaries about a client's profile
- Understand client's fleet of vehicles and tracking setup

**Returns**: Nested dictionary with:
- `client_info`: Personal details (name, ID, contact details)
- `addresses`: List of physical and postal addresses
- `vehicles`: List of all registered vehicles with details
- `sim_card`: SIM card information
- `fitment`: Installation details
- `billing`: Subscription information

### `get_client_vehicle_info(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves all vehicles belonging to a client.

**Returns**: List of dictionaries with vehicle information:
- `vehicle_id`, `registration`, `make`, `model`, `color`
- `chassis_number`, `model_year`, `contract_status`
- `terminal_serial`, `terminal_last_response`

### `client_call_verification(user_id: str) -> Dict[str, Any]`
**Purpose**: Retrieve a verification token for client call authentication.

**Returns**: Dictionary with either `token` or `error` key

## **Account Status & Financial Tools**

### `get_debtor_age_analysis(...) -> List[Dict[str, Any]]`
**Purpose**: Retrieves detailed financial aging analysis for debtors/clients with outstanding balances.

**Parameters**:
- `number_records`, `user_id`, `client_type`, `min_age_days`, `max_age_days`
- `min_balance_total`, `invoice_option`, `payment_arrangement_status`
- `pre_legal`, `rejected`, `sort_by`, `sort_direction`

**Use Cases**: Analyze accounts receivable, outstanding balances across aging periods

### `get_client_account_aging(user_id: str) -> Optional[List[Dict[str, str]]]`
**Purpose**: Retrieves aging analysis of client's outstanding account balance.

**Returns**: Dictionary with aging categories:
- `x0`: Current amount due, `x30`: 1-30 days overdue
- `x60`: 31-60 days overdue, `x90`: 61-90 days overdue
- `x120`: 91+ days overdue, `xbalance`: Total outstanding

### `get_client_account_status(user_id: str) -> Optional[List[Dict[str, str]]]`
**Purpose**: Checks if client account is in good standing or has payment issues.

**Returns**: Dictionary with `account_state` indicating standing

### `get_client_account_overview(user_id: str) -> Optional[Dict[str, str]]`
**Purpose**: Retrieves 360° dashboard view of client account status and health.

**Returns**: Comprehensive account status across multiple dimensions:
- Financial health, service information, communication status
- Account management, payment parameters, risk indicators

### `get_client_billing_analysis(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves detailed financial and billing analysis for a client account.

**Returns**: Comprehensive account metrics including payment status, aging, financial metrics

## **Payment & Banking Tools**

### `get_client_banking_details(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves comprehensive banking details for a client.

**Returns**: Complete banking and payment information including:
- Account details, payment configuration, billing preferences
- Account status, bank information, payment methods

### `get_client_payment_history(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves chronological history of payment promises/commitments and their outcomes.

**Returns**: List of payment commitments with:
- `arrangement_id`, `amount`, `pay_date`, `arrangement_state`
- `created_by`, fulfillment status

### `get_client_failed_payments(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves history of failed debit order payments.

**Returns**: List of failed payments with `payment_date` and `failure_reason`

### `get_client_last_successful_payment(user_id: str) -> Dict[str, Any]`
**Purpose**: Retrieves information about the most recent successful payment.

**Returns**: `payment_id`, `payment_date`, `payment_amount` or None if no payments found

### `get_client_last_valid_payment(user_id: str) -> Dict[str, Any]`
**Purpose**: Retrieves most recent valid payment, excluding reversed payments.

### `get_client_last_reversed_payment(user_id: str) -> Dict[str, Any]`
**Purpose**: Retrieves most recent payment that was later reversed.

## **Payment Arrangement Creation Tools**

### `create_payment_arrangement(...) -> dict`
**Purpose**: Creates a payment arrangement with up to three installments.

**Parameters**:
- `user_id`, `pay_type_id`, `payment1`, `date1`, `note`
- `payment2`, `date2`, `payment3`, `date3`
- `mandate_id1`, `mandate_fee`

**Payment Types**: 1=Direct Debit, 2=EFT, 3=Credit Card, 4=OZOW, 5=Pay@, 6=Cheque, 7=Capitec Pay

**Returns**: Success status, message, `arrangement_id`

### `create_payment_arrangement_payment_portal(...) -> dict`
**Purpose**: Creates a single-payment arrangement through the payment portal.

**Parameters**: `user_id`, `payment_type_id`, `payment_date`, `amount`, `online_payment_reference_id`

### `create_debicheck_payment(...) -> dict`
**Purpose**: Creates a DebiCheck payment arrangement (default: one-time payment).

**Parameters**: Multiple payment options with mandate creation and fees

### `generate_sms_payment_url(...) -> Dict[str, Any]`
**Purpose**: Generates a secure payment URL for client online payments.

**Parameters**: `user_id`, `amount`, `optional_reference`

**Returns**: Success status, payment URL, reference ID

## **DebiCheck & Mandate Tools**

### `get_client_debit_mandates(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves electronic debit order authorizations (DebiCheck mandates).

**Returns**: List of mandates with status, amounts, collection details, account information

### `create_mandate(...) -> dict`
**Purpose**: Creates a debit order mandate (recurring or once-off).

**Parameters**: `user_id`, `service`, `amount`, `collection_date`, `authentication_code`

**Returns**: Success status, mandate details, mandate type

### `get_payment_arrangement_types(query: str = "all") -> List[Dict[str, str]]`
**Purpose**: Retrieves payment arrangement types available for customers.

## **Contract & Subscription Tools**

### `get_client_contracts(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves all vehicle tracking contracts for a client.

**Returns**: Contract information including status, dates, payment options, categories

### `get_client_subscription_amount(...) -> Dict[str, Any]`
**Purpose**: Calculates client's total subscription amount across all contracts.

**Parameters**: `user_id`, optional `start_date`, `end_date`

**Returns**: Total subscription amount and calculation period

## **Communication & Notes Tools**

### `get_client_notes(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves client notes and communication history.

**Returns**: List of communications with timestamp, note content, source, created_by

### `add_client_note(user_id: str, note_text: str) -> Dict[str, Any]`
**Purpose**: Adds a note to client's record with current timestamp.

**Returns**: Success status, confirmation message, `client_note_id`

### `get_client_debtor_letters(user_id: str) -> List[Dict[str, Any]]`
**Purpose**: Retrieves history of automated letters sent to client from Debtors department.

**Returns**: List of letters with report details and sent dates

## **Client Information Update Tools**

### `update_client_contact_number(user_id: str, mobile_number: str) -> dict`
**Purpose**: Updates the direct contact number (mobile number) for a client.

**Returns**: Success status, previous and current numbers

### `update_client_email(user_id: str, email_address: str) -> dict`
**Purpose**: Updates the email address for a client.

**Returns**: Success status, previous and current email addresses

### `update_client_next_of_kin(...) -> dict`
**Purpose**: Updates next of kin information for a client.

**Parameters**: `user_id`, `nok_name`, `nok_phone`, `nok_email`

### `update_client_banking_details(user_id: str, **update_fields) -> dict`
**Purpose**: Updates banking details for a specific client.

**Parameters**: Various banking fields like account details, invoice options, etc.

## **Validation Tools**

### `validate_next_of_kin_contact(...) -> Dict[str, Any]`
**Purpose**: Validates next of kin's contact information to prevent duplicates.

**Parameters**: `user_id`, `name`, `phone`, `email`

**Returns**: Validation result with conflict information

### `validate_next_of_kin_emergency_contact(...) -> Dict[str, Any]`
**Purpose**: Validates vehicle emergency contact to prevent duplicates with existing next of kin.

## **Call Disposition & Management Tools**

### `get_disposition_types(include_disabled: bool = False) -> dict`
**Purpose**: Retrieves all available call disposition types.

**Returns**: List of disposition types with descriptions and instructions

### `save_call_disposition(...) -> dict`
**Purpose**: Saves call disposition and optional note for a debtor call.

**Parameters**: `client_id`, `disposition_type_id`, `rating`, `ratio`, `note_text`

### `update_payment_arrangements(user_id: str) -> Dict[str, Any]`
**Purpose**: Updates payment arrangement statuses for a client.

## **Reference Data Tools**

### `get_bank_options() -> dict`
**Purpose**: Retrieves all available banking options for client banking operations.

**Returns**: Banks, account types, categories, invoice options, debit run days, PACS services

### `date_helper(query=None)`
**Purpose**: Simple date helper that accepts natural language date queries.

**Returns**: Date information including ISO date, weekday, human-readable description

### `get_current_date_time()`
**Purpose**: Returns the current date and time as a string.

## **Usage Guidelines for Agent Development**

1. **Always use user_id as string** when calling these tools
2. **Handle errors gracefully** - most tools return error information in the response
3. **Check success status** in returned dictionaries before proceeding
4. **Use appropriate tools for verification** before allowing account discussions
5. **Follow payment creation hierarchy**: Mandate → Arrangement → Portal
6. **Update client information** after successful interactions
7. **Add notes** to document all client interactions
8. **Save dispositions** at the end of calls

## **Common Integration Patterns**

```python
# Basic client lookup pattern
profile = get_client_profile.invoke(user_id)
account_overview = get_client_account_overview.invoke(user_id)
account_aging = get_client_account_aging.invoke(user_id)

# Payment arrangement pattern  
mandate_result = create_mandate.invoke({...})
if mandate_result["success"]:
    arrangement_result = create_payment_arrangement.invoke({...})

# Client update pattern
update_result = update_client_contact_number.invoke({...})
if update_result["success"]:
    add_client_note.invoke({"user_id": user_id, "note_text": "Contact updated"})
```

This comprehensive toolset enables full debt collection call functionality including client verification, account analysis, payment processing, and relationship management.