"""Simple observable-feedback demo, not an optimized competition strategy."""
import math


def run(client, problem=4):
    """Serpentine discovery plus bearing following; uses no scene truth."""
    client.enter()
    cleared=set()
    measurements=0
    try:
        # Every target is inside a 500 m grid cell whose four vertices are <=708 m away.
        # This supplies discovery locations on both sides of a directional source.
        for iy,y in enumerate(range(-2000,2001,500)):
            xs=list(range(-2000,2001,500))
            if iy%2: xs.reverse()
            for x in xs:
                for channel in range(1,21):
                    if channel in cleared: continue
                    observation=client.measure((x,y),channel)
                    measurements+=1
                    position=(x,y)
                    step=200.0
                    previous=None
                    last_visible=None
                    for _ in range(100):
                        kind=observation['measure_result']
                        if kind=='no_signal':
                            if last_visible is None: break
                            # After crossing a directional lobe boundary, optical clear can
                            # still succeed. Otherwise bisect the last movement and retry.
                            if step<=12.5 and client.clear(position,channel)['clear_result']=='success':
                                cleared.add(channel); break
                            step/=2
                            position=((position[0]+last_visible[0])/2,(position[1]+last_visible[1])/2)
                            observation=client.measure(position,channel)
                            measurements+=1
                            continue
                        if kind=='near':
                            if client.clear(position,channel)['clear_result']=='success': cleared.add(channel)
                            break
                        bearing=observation['svd_deg']
                        if previous is not None and abs(math.remainder(bearing-previous,360))>90:
                            step/=2
                        if step<=12.5:
                            if client.clear(position,channel)['clear_result']=='success':
                                cleared.add(channel); break
                        angle=math.radians(bearing)
                        last_visible=position
                        position=(position[0]+step*math.cos(angle),position[1]+step*math.sin(angle))
                        previous=bearing
                        observation=client.measure(position,channel)
                        measurements+=1
                    if len(cleared)==16:
                        return dict(cleared_channels=sorted(cleared),measurement_count=measurements)
        return dict(cleared_channels=sorted(cleared),measurement_count=measurements)
    finally:
        if client.active: client.exit()
