scenario = "Trials and Templates 6";

default_font_size = 48;

begin;

text { caption = "red"; font_color = 255,0,0; } red;
text { caption = "green"; font_color = 0,255,0; } green;
text { caption = "blue"; font_color = 0,0,255; } blue;
text { caption = "flower"; } flower;
text { caption = "car"; } car;
text { caption = "house"; } house;

TEMPLATE "test.tem" {
    part1        code1        part2        code2        ;
    red         "red"         car          "car"        ;
    blue        "blue"        house        "house"      ;
    green       "green"       flower       "flower"     ;
};