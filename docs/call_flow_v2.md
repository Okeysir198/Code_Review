## Ratio 1 - Inflow Call Flow Visualization (Cleaned)

Here is a step-by-step visualization of the agent actions based on the call flow from the "Ratio 1 - Inflow.docx" document:

**1. Introduction**
   - Agent: Greet, state name & department (Cartrack Accounts), ask for the client by name ("May I speak to {Client’s Full Name}?").

**2. Name Verification**
   - Agent: Confirms if they are speaking to the correct person, e.g., "Are you {Client's Full Name}?" or listens to the initial response.
   - **Decision Point:** Outcome of Name Verification?
      - **Verified (Right Person):** Proceed to Step 3.
      - **3rd Party:**
         - Agent: Delivers message asking 3rd party to have client call 011 250 3000.
         - *[End Call Flow Branch]*
      - **Wrong Person:**
         - Agent: Apologize for the error, state you have the wrong number.
         - *[End Call Flow Branch]*
      - **Client Unavailable:**
         - Agent: Ask the person to advise the client Cartrack called and to call back 011 250 3000, or attempt to schedule a callback.
         - *[End Call Flow Branch]*
      - **Insufficient Info / Unclear:**
         - Agent: Re-ask or attempt to clarify if they are speaking to the intended client. If still unclear after attempts, may need to end the call politely.

**3. Details Verification**
   - Agent: State call is recorded for quality/security ("Please note that this call is recorded for quality and security purposes...").
   - Agent: Request verification of specific details ("...and to ensure I am speaking to the right person, please confirm the following details:").
   - **Verification Requirement:** Agent needs the client to confirm **ANY 3** of the following items **OR** confirm the **Full ID number alone** **OR** confirm the **Full Passport number alone**:
      - Username
      - Date of birth
      - Vehicle registration
      - Vehicle Make
      - Vehicle Model
      - Vehicle colour
      - VIN Number
      - Email address
      - Residential Address
      - Contact Number
      - Full ID number *(Confirmation of this item alone is sufficient)*
      - Full Passport number *(Confirmation of this item alone is sufficient)*
   - *(Agent proceeds only after successful verification)*

**4. Reason for Call**
   - Agent: Inform client payment wasn't received, account is overdue.
   - Agent: State the immediate payment amount required.

**5. Negotiation**
   - Agent: Explain the **Consequences** of not making an immediate payment:
      - *(i) Suspension of access to the Cartrack App.*
      - *(ii) Suspension of vehicle positioning through the control room.*
      - *(iii) Suspension of notifications regarding the vehicle.*
   - Agent: Explain the **Benefits** of paying the account:
      - *"Once the payment has been received, your account will be in good standing and all suspended services will be reinstated."*
   - Agent: **Resolve any client queries effectively.**

**6. Promise to Pay (PTP)**
   - Agent: Ask, "Can we debit this amount from your account today?"
   - **Decision Point:** Client's response?
      - **YES:**
         - Agent: Load immediate Debit Order in the system (e.g., CAMS).
         - Proceed to Step 8.
      - **NO:**
         - Agent: Initiate loading a Debicheck PTP.
         - Agent: Ask for bank details (salary account) & confirm salary date.
         - Agent: Address any history of failed/reversed debits.
         - Agent: If bank details changed, update CAMS first.
         - Agent: Load Debicheck PTP in the system.
         - Proceed to Step 7.

**7. Debicheck Explanation**
   - Agent: Explain the Debicheck authentication process (via bank app/SMS).
   - Agent: Instruct client to approve the request promptly.
   - Agent: Inform client about the R10 fee and total debit amount.
   - Agent: Check/fix bank details in the system before finalizing.
   - Proceed to Step 8.

**8. Subscription Reminder**
   - Agent: Remind client the payment is for arrears; normal subscription is separate.
   - Proceed to Step 10.

**--- Alternative Payment Path ---**

**9. Cartrack Payment Portal**
   - Agent: Inform client SMS link was sent (or send it via CAMS if not received).
   - Agent: Guide client on using the link:
      - Click the "PAY HERE" button in the SMS.
      - Edit the payment amount if necessary to match the agreed amount.
      - Click "CONFIRM & NEXT".
      - Select the preferred payment method: Card Payment, Ozow, CapitecPay, or Pay@.
   - Agent: Guide the client through the specific steps for their chosen method.
   - Agent: Explain the post-payment process shown on the portal.
   - Proceed to Step 10.

**--- Resume Main Flow ---**

**10. Client Details Update**
    - Agent: Update Contact Number, Email, Banking Details, Next of Kin in the system (if needed).

**11. Referrals**
    - Agent: Ask about referrals & explain the benefit (2 months free).
    - **Decision Point:** Does the client have referrals?
       - **YES:** Agent asks for contact details & actions referral in the portal.
       - **NO:** Proceed to Step 12.

**12. Further Assistance**
    - Agent: Ask if the client needs any other help with their account.
    - **Decision Point:** Does the client need more help?
       - **YES:** Agent assists accordingly.
       - **NO:** Proceed to Step 13.

**13. Add Detailed Note**
    - Agent: Add comprehensive notes about the call interaction into the system.

**14. Disposition Correctly**
    - Agent: Select the correct call outcome code in the system.
    - *[End Standard Call Flow]*

**--- Conditional Flow (Client Initiated) ---**

**15. Cancellation Declaration**
    - Agent: Establish reason and contract details.
    - Agent: Advise client of cancellation value/total balance.
    - Agent: Inform client the request will be escalated to Client Services.
    - Agent: Log a cancellation ticket on the Helpdesk.
    - Agent: Deliver script confirming fee, balance, escalation, and next steps.
    - *[End Cancellation Call Flow]*
