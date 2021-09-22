#include "mbed.h"
#include "platform/mbed_thread.h"
#include "ESP8266Interface.h"
#include <MQTTClientMbedOs.h>



int AInScaled = 0;
int sampleCount = 0;
char buffer[128];
char buffer2[128];
char buffer3[128];

double slip;
double FrontMstoKmh = 0.0; //= (int)speedFront * 3.6;
double RearMstoKmh = 0.0; //= (int)speedRear * 3.6;

Timer t;
Timer r;
InterruptIn risingEdgeFront(PA_12);
InterruptIn risingEdgeRear (PA_11);

AnalogIn ain5(A5);
DigitalOut myled1(PB_5);
DigitalOut myled2(PB_0);
DigitalOut TCSled(PB_1);
int slipModeOne = 6;
int slipModeTwo = 15;
int TCSmode;



 
volatile long int counttOne;
volatile long int counttTwo;

void pulsesFront() {
    if(myled2 == 1) {
        myled2 = 0;
    } else {
        myled2 = 1;
    }
    counttOne++;
}
void pulsesRear() {
    if(myled1 == 1) {
        myled1 = 0;
    } else {
        myled1 = 1;
    }
    counttTwo++;
}
 void simultaion() {
        static int count = 200;
        static double upone = 0.7;
        static double uptwo = 0.8;
        count = count + 82;
        printf("count %d\n", count);
        FrontMstoKmh += upone;           // FrontMstoKmh
        RearMstoKmh += uptwo;           // RearMstoKmh
        //printf("front %d\n", (int)x);
        //printf("rear %d\n", (int)y);
        
        if (count > 4095){
            count = 200;
            FrontMstoKmh = 0.0;
            RearMstoKmh = 0.0;
        }
        ThisThread::sleep_for(600ms);
    }
void slipper() {
    double sliperi = 0.0;
    sliperi = RearMstoKmh / FrontMstoKmh;
    
    sliperi = sliperi -1;
    
    sliperi = sliperi * 100;
    
    slip = sliperi;
    
}
    
int main() {
    
    ESP8266Interface esp(MBED_CONF_APP_ESP_TX_PIN, MBED_CONF_APP_ESP_RX_PIN);
    
    //Store device IP
    SocketAddress deviceIP;
    //Store broker IP
    SocketAddress MQTTBroker;
    
    TCPSocket socket;
    MQTTClient client(&socket);
    
    ThisThread::sleep_for(3s);    
    
    printf("\nConnecting wifi..\n");

    int ret = esp.connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);

    if(ret != 0)
    {
        printf("\nConnection error\n");
    }
    else
    {
        printf("\nConnection success\n");
    }
        
    esp.get_ip_address(&deviceIP);
    printf("IP via DHCP: %s\n", deviceIP.get_ip_address());

    
    esp.gethostbyname(MBED_CONF_APP_MQTT_BROKER_HOSTNAME, &MQTTBroker, NSAPI_IPv4, "esp");

    MQTTBroker.set_port(MBED_CONF_APP_MQTT_BROKER_PORT);

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;       
    data.MQTTVersion = 3;
    
    data.clientID.cstring = MBED_CONF_APP_MQTT_CLIENT_ID;
    data.username.cstring = MBED_CONF_APP_MQTT_AUTH_METHOD;
    data.password.cstring = MBED_CONF_APP_MQTT_AUTH_TOKEN;

    
    sprintf(buffer, "Hello from Mbed OS %d.%d", MBED_MAJOR_VERSION, MBED_MINOR_VERSION);
                    
    MQTT::Message msg;
    msg.qos = MQTT::QOS0;
    msg.retained = false;
    msg.dup = false;
    msg.payload = (void*)buffer;
    msg.payloadlen = strlen(buffer);
    
    
    ThisThread::sleep_for(5s);  
                                
                                
    // Connecting mqtt broker
    printf("Connecting %s ...\n", MBED_CONF_APP_MQTT_BROKER_HOSTNAME);
    socket.open(&esp);
    socket.connect(MQTTBroker);
    client.connect(data);
    
    
    
    //Publish                    
    printf("Publishing with payload length %d\n", strlen(buffer));
    client.publish(MBED_CONF_APP_MQTT_TOPIC, msg);
    
    
    
    
    
    risingEdgeFront.rise(&pulsesFront); 
    risingEdgeRear.rise(&pulsesRear);
    while(1) {
        t.reset();
        r.reset();
        t.start();
        r.start();
        counttOne = 0;
        counttTwo = 0;
        while(t.read_ms() < 1001) {
            ;
        }
        while(r.read_ms() < 1001) {
            ;
        }     
        t.stop();
        r.stop();
        long int tempOne = counttOne;
        long int tempTwo = counttTwo;
        //printf("Count: %d", temp);
        double circumferenceFront = 0.06 * 3.1416; 
        double circumferenceRear = 0.06 * 3.1416; 
        
        double revFront = (double)tempOne;
        double rpmFront = revFront*60;
        double speedFront = circumferenceFront * revFront;  
        
        double revRear = (double)tempOne;
        double rpmRear = revRear*60;
        double speedRear = circumferenceRear * revRear; 
        
        //double FrontMstoKmh; //= (int)speedFront * 3.6;
        //double RearMstoKmh; //= (int)speedRear * 3.6;
        
        simultaion();
        
        slipper();
        
        
        //printf(" %0.2f RPM", rpm);
        //printf("front speed:  %d km/h\n" , (int)FrontMstoKmh);
        //printf("rear speed:  %d km/h\n" , (int)RearMstoKmh);
        //printf("slip: %d %\n", (int)slip);
        
        //while(1)  { // This only for testing. When system mounted car, this is just while(1)....
        AInScaled = ain5.read_u16() >> 4;
                                           
            //sampleCount += 1;    
            // {\"d\":{\"Sensors\":\"N1ER \",\"SampleNr\":%d,\"slip \":%d,\"rear speed \":%d,\"front speed \":%d}}                                 
        sprintf(buffer, "{\"d\":{\"Sensors\":\"N1ER \",\"SLIP\":%d,\"Rear speed\":%d,\"Front speed\":%d,\"TCS mode\":%d}} ",(int)slip,(int)RearMstoKmh,(int)FrontMstoKmh,TCSmode);
        msg.payload = (void*)buffer;
        msg.payloadlen = strlen(buffer);
            
            //Publish                    
        printf("Publishing with payload length %d\n", strlen(buffer));
        client.publish("iot-2/evt/Sensor/fmt/json", msg);
        printf("front speed:  %d km/h\n" , (int)FrontMstoKmh);
        printf("rear speed:  %d km/h\n" , (int)RearMstoKmh);
        printf("slip: %d %\n", (int)slip);
            
        ThisThread::sleep_for(1s);  
            
        if(AInScaled > 1900 && AInScaled < 1960 && slip > slipModeOne){
            printf("TCS decrease motor power...\n");
                //TCSled = 1; Maybe led blinks when tcs decrease
            
        }
        if(AInScaled > 3900 && AInScaled > 4000 && slip > slipModeTwo ){
            printf("TCS decrease motor power...n");
                //TCSled =1; Maybe led blinks when tcs decrease
            
        }
        if(AInScaled > 1900 && AInScaled < 2000){
             TCSmode = 1;
        }
        else{
             TCSmode = 2;
        }    
        
        
        // This only for testing. When system mounted car, this coming when shutting down system...
        //printf("Disconnecting from MQTT broker");
        //client.disconnect();
        //ThisThread::sleep_for(2s);
        
        //socket.close();
        //printf("Entering deepsleep (press RESET button to resume)\n"); 
        //ThisThread::sleep_for(300s);
    }
}