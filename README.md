# cricket-trajectory-tracer
Tracing the trajectory of a cricket ball with zone and speed detection.
The script detects the bowler's action (the angle between the bowling arm's wrist, shoulder, and back being 180 degrees) and clips the next 3-4 seconds. This is later run through YOLO with SAHI, and all further processes(ball tracking, speed detection, and zone detection) are done.
Speed detection - The number of frames when the ball is released from the bowler to reach the batsman is stored and then the speed is calculated.
Zone detection - The script initially asks you to annotate the endpoints of the pitch and then the zones are mapped out. For the ball detection frame with the lowest y-coordinates, the zone is calculated.
Examples:



https://github.com/BHUVAN-RJ/cricket-trajectory-tracer/assets/83565908/73b11238-d454-4723-ae58-005c4a17b04f

