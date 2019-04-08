function q3 = qmult(q2, q1)

% performs quaternion multiplication on two length
% four vectors, with the fourth element being
% the scalar element
% outputs q3 as a vertical vector
% inputs should be vertical vectors


q3s = q1(4)*q2(4) - dot(q2(1:3), q1(1:3));
q3v = q1(4)*q2(1:3) + q2(4)*q1(1:3) - cross(q2(1:3), q1(1:3));

q3 = [q3v(1); q3v(2); q3v(3); q3s];
