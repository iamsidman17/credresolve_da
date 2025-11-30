import pandas as pd
import numpy as np
import os
import json

DATA_DIR = '/Users/sidharthmanakil/Documents/IITM/Placements/CredResolve_DA/data'

def load_data():
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    meta = pd.read_csv(os.path.join(DATA_DIR, 'metaData.csv'))
    
    calls = pd.read_csv(os.path.join(DATA_DIR, 'call_placed.csv'))
    sms = pd.read_csv(os.path.join(DATA_DIR, 'AI_sms_callback.csv'))
    whatsapp = pd.read_csv(os.path.join(DATA_DIR, 'whatsapp_activity.csv'))
    visits = pd.read_csv(os.path.join(DATA_DIR, 'mobile_app_data.csv'))
    teleco = pd.read_csv(os.path.join(DATA_DIR, 'teleco_call_back.csv'))
    
    return train, test, meta, calls, sms, whatsapp, visits, teleco

def process_interactions(calls, sms, whatsapp, visits, teleco):
    print("Processing interactions...")
    
    # Process Calls (Human)
    calls_agg = calls.groupby('lead_code').agg(
        total_calls=('duration', 'count'),
        avg_call_duration=('duration', 'mean'),
        answered_calls=('disposition', lambda x: (x == 'ANSWERED').sum()),
        no_answer_calls=('disposition', lambda x: (x == 'NO_ANSWER').sum()),
        busy_calls=('disposition', lambda x: (x == 'BUSY').sum())
    ).reset_index()
    
    # Process Teleco (Bot) - Extract Intent/Sentiment
    def parse_transcript(x):
        try:
            return json.loads(x)
        except:
            return {}

    teleco['transcript_dict'] = teleco['transcript_json'].apply(parse_transcript)
    teleco['sentiment'] = teleco['transcript_dict'].apply(lambda x: x.get('sentiment', 'UNKNOWN'))
    
    teleco_agg = teleco.groupby('lead_code').agg(
        bot_duration_mean=('duration', 'mean'),
        bot_answered=('disposition', lambda x: (x == 'ANSWERED').sum()),
        bot_busy=('disposition', lambda x: (x == 'BUSY').sum()),
        bot_no_answer=('disposition', lambda x: (x == 'NO_ANSWER').sum()),
        bot_sentiment_pos=('sentiment', lambda x: (x == 'positive').sum())
    ).reset_index()
    
    # Process SMS - Dropping as per user request (implied "practical" list)
    # Keeping it minimal if not requested, but user didn't explicitly say "drop SMS". 
    # However, "my opinion is this: 1)... 2)... 3)... 4)..." suggests an exclusive list.
    # I will keep SMS basic counts just in case, but prioritize the others.
    sms_agg = sms.groupby('lead_code').agg(
        total_sms=('status', 'count')
    ).reset_index()
    
    # Process WhatsApp
    whatsapp['is_pay_link'] = whatsapp['response_message'].str.contains('link', case=False, na=False)
    
    whatsapp_agg = whatsapp.groupby('lead_code').agg(
        wa_pay_link=('is_pay_link', 'max') # 1 if present, 0 if not
    ).reset_index()
    
    # Process Visits
    visits_agg = visits.groupby('lead_code').agg(
        visit_met_customer=('result', lambda x: (x == 'MET_CUSTOMER').sum()),
        visit_door_locked=('result', lambda x: (x == 'DOOR_LOCKED').sum()),
        visit_shifted=('result', lambda x: (x == 'SHIFTED').sum())
    ).reset_index()
    
    return calls_agg, teleco_agg, sms_agg, whatsapp_agg, visits_agg

def merge_features(df, meta, calls_agg, teleco_agg, sms_agg, whatsapp_agg, visits_agg):
    print("Merging features...")
    
    # Merge Metadata
    df = df.merge(meta, on='lead_code', how='left')
    
    # Merge Interactions
    df = df.merge(calls_agg, on='lead_code', how='left')
    df = df.merge(teleco_agg, on='lead_code', how='left')
    df = df.merge(sms_agg, on='lead_code', how='left')
    df = df.merge(whatsapp_agg, on='lead_code', how='left')
    df = df.merge(visits_agg, on='lead_code', how='left')
    
    # Fill missing values for interaction counts with 0
    interaction_cols = [
        'total_calls', 'avg_call_duration', 'answered_calls', 'no_answer_calls', 'busy_calls',
        'bot_duration_mean', 'bot_answered', 'bot_busy', 'bot_no_answer', 'bot_sentiment_pos',
        'total_sms',
        'wa_pay_link',
        'visit_met_customer', 'visit_door_locked', 'visit_shifted'
    ]
    df[interaction_cols] = df[interaction_cols].fillna(0)
    
    return df

def main():
    train, test, meta, calls, sms, whatsapp, visits, teleco = load_data()
    
    calls_agg, teleco_agg, sms_agg, whatsapp_agg, visits_agg = process_interactions(calls, sms, whatsapp, visits, teleco)
    
    print("Creating training set...")
    train_processed = merge_features(train, meta, calls_agg, teleco_agg, sms_agg, whatsapp_agg, visits_agg)
    
    print("Creating test set...")
    test_processed = merge_features(test, meta, calls_agg, teleco_agg, sms_agg, whatsapp_agg, visits_agg)
    
    # Save processed data
    train_processed.to_csv('train_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)
    print("Processed data saved to train_processed.csv and test_processed.csv")

if __name__ == "__main__":
    main()
