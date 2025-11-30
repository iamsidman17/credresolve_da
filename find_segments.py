import pandas as pd

def find_low_target_segments():
    train = pd.read_csv('train_processed.csv')
    
    # Segment 1: No Answered Calls
    seg1 = train[train['answered_calls'] == 0]
    print(f"Segment: Answered Calls == 0")
    print(f"Count: {len(seg1)}")
    print(f"Mean Target: {seg1['TARGET'].mean():.4f}")
    
    # Segment 2: No Answered Calls AND No Met Customer
    seg2 = train[(train['answered_calls'] == 0) & (train['met_customer'] == 0)]
    print(f"\nSegment: Answered Calls == 0 AND Met Customer == 0")
    print(f"Count: {len(seg2)}")
    print(f"Mean Target: {seg2['TARGET'].mean():.4f}")
    
    # Segment 3: No Interaction at all (No calls, no SMS delivered, no WA read, no visits)
    seg3 = train[
        (train['total_calls'] == 0) & 
        (train['sms_delivered'] == 0) & 
        (train['wa_read'] == 0) & 
        (train['met_customer'] == 0)
    ]
    print(f"\nSegment: No Interaction (Calls/SMS/WA/Visits)")
    print(f"Count: {len(seg3)}")
    print(f"Mean Target: {seg3['TARGET'].mean():.4f}")
    
    # Segment 4: Suggested Action = BOT and No Answered Calls
    seg4 = train[(train['suggested_action'] == 'ACTION_BOT') & (train['answered_calls'] == 0)]
    print(f"\nSegment: Action=BOT AND Answered Calls == 0")
    print(f"Count: {len(seg4)}")
    print(f"Mean Target: {seg4['TARGET'].mean():.4f}")

if __name__ == "__main__":
    find_low_target_segments()
