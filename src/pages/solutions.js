import React from "react"
import Layout from "../components/layout"
import SEO from "../components/seo"
import style from "./solutions.module.css"

const SolutionsPage = () => {
  return (
    <Layout>
      <SEO
        title="Our Solutions"
        description="ML+DSP Algorithm Solutions"
        image="/jumpML.svg"
        pathname="/solutions"
        // Boolean indicating whether this is an article:
        // article
      />
      <section className={style.wrapper}>
      <h2 className={style.heading}>Applications of JumpML NR</h2>

<li className={style.flexcontainer}>
<ul className={style.ul}>
 Voice Telephony 
<img
  src="markets/on_the_phone.svg"
  alt="Voice Telephony"
  className={style.center}
/>
<li className={style.flexitem}>Playback NR: Listen easy</li>
<li className={style.flexitem}>Mic NR: Be heard clearly</li>
<li className={style.flexitem}>Earbuds, headphones</li>
</ul>
<ul className={style.ul}>
  Augmented Hearing
<img
  src="markets/head_ear.svg"
  alt="SuperHearing"
  className={style.center}
/>
<li className={style.flexitem}>Hearing Aids</li>
<li className={style.flexitem}>Smart Transparency</li>
<li className={style.flexitem}>Smart ANC</li>   
</ul>
<ul className={style.ul}>
  Voice control/analytics
<img
  src="markets/CartoonRobot.svg"
  alt="Voice_Control"
  className={style.center}
/>
<li className={style.flexitem}>Noise robust VoiceUI</li>
<li className={style.flexitem}>Voice analytics front-end</li>
<li className={style.flexitem}>Battery-powered TV remote</li>
</ul>

<li className={style.flexitem}> <h1> JumpML Noise Reduction enables real-time voice and hearing applications anywhere/anytime and on-the-go.   </h1> </li>

</li>


 <h1 className={style.h1}> Demo 1: Outdoor Walk</h1>

<li className={style.flexcontainer}>
<ul className={style.ul}> 
 Windy
<img
  src="noises/wind-blowing-cloud.svg"
  alt="WindNoise"
  className={style.center}
/>
</ul>
<ul className={style.ul}> 
  Siren
<img
  src="noises/ambulance.svg"
  alt="SirenNoise"
  className={style.center}
/> 
</ul>
<ul className={style.ul}> 
  Birds
<img
  src="noises/crow.svg"
  alt="Crow"
  className={style.center}
/>
</ul>
</li>
<ul className={style.ul}> 
  <li className={style.flexitem}> Noisy input (before NR) </li>
  <audio controls src="audio/outdoor.wav"/>
  </ul>

  <ul className={style.ul}> 
  <li className={style.flexitem}> JumpML NR output  </li>
  <audio controls src="audio/outdoor_output.wav"/>
  </ul>
 
  <h1> Demo 2: Cafe/restaurant</h1>


<li className={style.flexcontainer}>
<ul className={style.ul}> 
 Babble/chatter
<img
  src="talk.svg"
  alt="babble"
  className={style.center}
/>
</ul>
<ul className={style.ul}> 
  Road/vehicle 
<img
  src="noises/car_traffic.svg"
  alt="vehicular"
  className={style.center}
/> 
</ul>
</li>
  <ul className={style.ul}> 
  <li className={style.flexitem}> Noisy input (before NR) </li>
  <audio controls src="audio/cafe_mix.wav"/>
  </ul>

  <ul className={style.ul}> 
  <li className={style.flexitem}> JumpML NR output  </li>
  <audio controls src="audio/cafe_output.wav"/>
  </ul>

  <h1 className={style.h1}> Demo 3: Street/Vehicular</h1>

<li className={style.flexcontainer}>
<ul className={style.ul}> 
  Car
<img
  src="noises/skoda_car.svg"
  alt="Car"
  className={style.center}
/> 
</ul>
<ul className={style.ul}> 
  Traffic
<img
  src="noises/car_traffic.svg"
  alt="Traffic"
  className={style.center}
/> 
</ul>
<ul className={style.ul}> 
 Windy
<img
  src="noises/wind-blowing-cloud.svg"
  alt="WindNoise"
  className={style.center}
/>
</ul>
</li>
<ul className={style.ul}> 
  <li className={style.flexitem}> Noisy input (before NR) </li>
  <audio controls src="audio/vehicular_mix.wav"/>
  </ul>

  <ul className={style.ul}> 
  <li className={style.flexitem}> JumpML NR output  </li>
  <audio controls src="audio/vehicular_output.wav"/>
  </ul>

  <h1 className={style.h1}> Demo 4: Indoor/Home</h1>

<li className={style.flexcontainer}>
<ul className={style.ul}> 
  Gardening
<img
  src="noises/leaf_blower.svg"
  alt="Gardener"
  className={style.center}
/> 
</ul>
<ul className={style.ul}> 
  Rain/Thunder
<img
  src="noises/thunder.svg"
  alt="Traffic"
  className={style.center}
/> 
</ul>
<ul className={style.ul}> 
 Miscellaneous
<img
  src="noises/wind-blowing-cloud.svg"
  alt="WindNoise"
  className={style.center}
/>
</ul>
</li>
<ul className={style.ul}> 
  <li className={style.flexitem}> Noisy input (before NR) </li>
  <audio controls src="audio/home_mix.wav"/>
  </ul>

  <ul className={style.ul}> 
  <li className={style.flexitem}> JumpML NR output  </li>
  <audio controls src="audio/home_output.wav"/>
  </ul>


      </section>
    </Layout>
  )
}

export default SolutionsPage
