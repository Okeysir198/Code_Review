# AI Agent Debt Collection Call Actions Summary

Based on the optimized system, here's a comprehensive summary of all actions the AI agent needs to perform during a debt collection call:

## **Pre-Call Setup**
- Load client data using `get_client_data(user_id)`
- Initialize conversation state with `ConversationState()`
- Determine appropriate script type based on account status
- Build parameters with `prepare_parameters()`

## **Step-by-Step Call Actions**

### **1. Introduction**
**Actions:**
- Greet professionally: "Good day"
- Identify self and company: "This is [Agent Name] from Cartrack Accounts Department"
- Request specific client: "May I speak to [Title] [Full Name], please?"
- **Handle Responses:**
  - If wrong person: Insist on speaking to target client
  - If screening: "It's regarding their Cartrack account"
  - Never mention debt/amounts until verified

### **2. Name Verification**
**Actions:**
- Confirm client identity: "Just to confirm I'm speaking with [Full Name]?"
- Track verification attempts (max 3)
- Update state: `state.name_verification_status = "VERIFIED"`
- **Handle Non-Target Responses:**
  - Third party: Provide callback message with urgency
  - Unavailable: Give callback number
  - Wrong person: End call politely
  - Failed verification: Direct to customer service

### **3. Details Verification**
**Actions:**
- State security notice: "This call is recorded for quality and security"
- Request ONE verification field at a time from available options:
  - ID number (sufficient alone) OR
  - 3 items from: username, vehicle details, email, etc.
- Track matched fields: `state.add_matched_field(field)`
- Update verification status when complete
- **Critical Rule:** NO account details until FULLY verified

### **4. Reason for Call**
**Actions:**
- Thank for verification
- State account status directly: "Your account is overdue"
- Specify exact amount: "[Outstanding Amount] is required"
- Create urgency: "Immediate payment needed"
- Prepare for shock/denial responses
- Use objection responses from `parameters['objection_responses']`

### **5. Negotiation**
**Actions:**
- Explain consequences of non-payment:
  - Service suspension (app, positioning, notifications)
  - Credit profile impact
  - Recovery fees (R25,000 if vehicle stolen)
  - Legal action (for pre-legal accounts)
- Highlight benefits of payment:
  - Immediate service restoration
  - Account protection
  - Avoid escalation
- **Handle Objections:**
  - Use emotional responses: `parameters['emotional_responses']`
  - Address specific objections: `parameters['objection_responses']`
  - Update conversation state: `state.add_objection(objection_type)`

### **6. Promise to Pay**
**Actions:**
- Secure payment arrangement in hierarchy order:
  1. **Immediate Debit:** "Can we debit [amount] today?"
  2. **DebiCheck:** Set up with R10 fee
  3. **Payment Portal:** Send secure link
- Get specific commitments: amount, date, method
- Confirm understanding: "So that's [amount] on [date] via [method]?"
- **No-Exit Rule:** Must secure SOME arrangement before ending
- Create payment using appropriate tools:
  - `create_debicheck_payment()` for DebiCheck
  - `create_payment_arrangement_payment_portal()` for portal

### **7. DebiCheck Setup** (if applicable)
**Actions:**
- Explain authentication process
- Set expectations: "Bank will send authentication request"
- Confirm total amount including R10 fee
- Ensure client commits to approving request
- Use `create_mandate()` if needed

### **8. Payment Portal** (if applicable)
**Actions:**
- Guide through portal step-by-step
- Stay on call during payment process
- Generate URL using `generate_sms_payment_url()`
- Provide real-time assistance
- Confirm successful completion

### **9. Subscription Reminder**
**Actions:**
- Clarify difference between arrears and subscription
- Explain: "Today's payment covers arrears of [amount]"
- Remind: "Regular subscription of [amount] continues on [date]"
- Prevent confusion about double charging

### **10. Client Details Update**
**Actions:**
- Verify/update contact information:
  - Mobile number: `update_client_contact_number()`
  - Email: `update_client_email()`
  - Next of kin: `update_client_next_of_kin()`
- Frame as routine maintenance
- Update banking details if needed: `update_client_banking_details()`

### **11. Referrals**
**Actions:**
- Mention referral program briefly
- Explain 2-month free subscription benefit
- Collect details if interested
- Don't pressure if not interested

### **12. Further Assistance**
**Actions:**
- Ask: "Anything else regarding your account?"
- Address any additional concerns
- Resolve queries completely before closing

### **13. Closing**
**Actions:**
- Summarize key outcomes:
  - Payment secured/arranged
  - Services restored/protected
  - Next steps confirmed
- Thank for cooperation
- Professional farewell
- Document call outcome

## **Throughout Call - Continuous Actions**

### **State Management**
- Update emotional state: `state.update_emotional_state()`
- Track objections: `state.add_objection()`
- Monitor payment willingness: `state.update_payment_willingness()`
- Adjust approach based on client responses

### **Behavioral Intelligence**
- Adapt tone to client's emotional state
- Use appropriate urgency level based on account risk
- Apply tactical guidance from behavioral analysis
- Escalate using `parameters['escalation_responses']` if needed

### **Documentation**
- Add detailed call notes: `add_client_note()`
- Save call disposition: `save_call_disposition()`
- Update payment arrangements: `update_payment_arrangements()`

## **Special Scenarios**

### **Cancellation Request**
**Actions:**
- Acknowledge request professionally
- Explain cancellation fees
- State total balance required
- Create cancellation ticket
- Set proper expectations

### **Query Resolution**
**Actions:**
- Address question thoroughly
- Confirm understanding
- Redirect to payment resolution
- Maintain call momentum

### **Escalation**
**Actions:**
- Acknowledge concern
- Create appropriate ticket
- Provide timeline and reference number
- Set realistic expectations

## **Success Metrics to Track**
- Payment arrangement secured (target: 75%+)
- Call completion rate (target: 90%+)
- Objection resolution (target: 80%+)
- Client satisfaction maintained
- Compliance with all verification requirements

## **Critical Rules Always Apply**
1. **No account details until fully verified**
2. **Professional tone throughout**
3. **Every call must end with some commitment**
4. **Document everything properly**
5. **Follow escalation procedures when needed**
6. **Maintain compliance with POPI and legal requirements**

This comprehensive action plan ensures the AI agent conducts effective, compliant, and professional debt collection calls while maximizing payment recovery rates.