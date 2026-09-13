"""Minimal custom batch-policy entry. Replace these actions with your strategy."""

def run(client, problem):
    client.enter()
    try:
        for channel in range(1,21):
            result=client.measure((0,0),channel)
            if result['measure_result']=='near':
                client.clear((0,0),channel)
            # result['svd_deg'] exists only when measure_result == 'direction'.
    finally:
        client.exit()
