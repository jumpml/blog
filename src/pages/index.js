// import React from "react"
// import Layout from "../components/layout"

// import style from "./index.module.css"

// const IndexPage = () => {
//   return (
//     <Layout>
//       <section className={style.wrapper}>
//         <p>Coming Soon!</p>
//         <p>We will have lots of articles on several interesting topics.</p>
//       </section>
//     </Layout>
//   )
// }

// export default IndexPage

import React from "react"
// import { graphql } from "gatsby"
// import PostLink from "../components/post-link"
import Layout from "../components/layout"
import SEO from "../components/seo"
import { Link } from "gatsby"
import style from "./index.module.css"


const IndexPage = (//{
  //data: {
    // allMarkdownRemark: { edges },
  //},
//}
) => {

  
  // const Posts = edges
  //   .filter(edge => !!edge.node.frontmatter.date) // You can filter your posts based on some criteria
  //   .map(edge => <PostLink key={edge.node.id} post={edge.node} />)

  // return <div>{Posts}</div>

  return (


    <Layout>
      <SEO
        title="JumpML. Learn and build useful things!"
        description="JumpML Website"
        image="/jumpML.svg"
        pathname="/"
        // Boolean indicating whether this is an article:
        // article
      />
      {<section className={style.wrapper}>

      <p>
          <i>
          Deep Learning is taking the world by storm, but 
          too complex to run on embedded devices. 
          </i>
          </p>
          <li className={style.flexcontainer}>
    
          <ul className={style.ul}>
          <img
            src="rocket.svg"
            width="366"
            height="374"
            alt="Low Latency"
            className={style.site_logo}
          />
            
            </ul>

          <ul className={style.ul}>
     
          JumpML Value Proposition

          <li className={style.flexitem}> <h1> Bring magical AI/ML experiences to embedded devices  </h1> </li>
   
          </ul>      
          </li>

          <li className={style.flexcontainer}>
        <ul className={style.ul}>
          <li className={style.flexcontainer}> Why bother with AI/ML on Embedded Devices? </li>  
          </ul>
          <ul className={style.ul}>
            Low-latency
          <img
            src="low-latency.svg"
            width="366"
            height="374"
            alt="Low Latency"
            className={style.site_logo}
          />
          </ul>
          <ul className={style.ul}>
            Privacy
            <img
            src="secured.svg"
            width="366"
            height="374"
            alt="Secure"
            className={style.site_logo}
          />
          </ul>
          <ul className={style.ul}>
            Energy-efficient
            <img
            src="energy-efficient.svg"
            alt="Green"
            width="366"
            height="374"
            className={style.site_logo}
          />
          </ul>
          <ul className={style.ul}>
            No Cloud Fee
            <img
            src="cash.svg"
            alt="Cash"
            width="366"
            height="374"
            className={style.site_logo}
          />
          </ul>

          
        </li>
          Embedded devices are severely 
          constrained in compute capability, size of memory and a 
          complicated path to model deployment. 
          <li className={style.flexcontainer}>
          <ul className={style.ul}>
          <li className={style.flexcontainer}>Challenges on Constrained Embedded Devices </li>  
          </ul>
          <ul className={style.ul}>
            Compute
          <img
            src="pgb-dsp.svg"
            width="366"
            height="374"
            alt="DSP"
            className={style.site_logo}
          />
          </ul>
          <ul className={style.ul}>
            Memory
            <img
            src="pgb-ram.svg"
            width="366"
            height="374"
            alt="Memory"
            className={style.site_logo}
          />
          </ul>
          <ul className={style.ul}>
            Model Deployment
            <img
            src="gears.svg"
            alt="Memory"
            width="366"
            height="374"
            className={style.site_logo}
          />
          </ul>
        </li>

        <p>

        At JumpML, we combine the simplicity and efficiency of classical digital signal processing (DSP) 
        and the best ML methods to develop high-performance solutions, 
        that are lightweight and energy-efficient. 

        </p>
        

        <p>
        Our internal development process involves training models in PyTorch, 
        followed by conversion of the models to C code, that can run 
        efficiently on any embedded system. 

        </p>

        <h1>More Information</h1>
        <p>
        Please check out our <Link to="/products">
        Products
              </Link>{" "} page for more information on our current offerings. 
        </p>
        <p>
        For technology demos and specific use cases, please check out our <Link to="/solutions">
        Use Cases</Link>{" "} page.
        </p>

        <p>
          For questions and more details, please contact us at{" "}
          <a href="mailto:info@jumpml.com">JumpML email.</a>
        </p>
        
        {/* <ul>{Posts}</ul> */}
      </section> }
    </Layout>
  )
}
export default IndexPage
// export const pageQuery = graphql`
//   query MyQuery {
//     allMarkdownRemark(sort: { order: DESC, fields: frontmatter___date }) {
//       edges {
//         node {
//           excerpt
//           id
//           fields {
//             slug
//           }
//           frontmatter {
//             title
//             date
//             subject
//             author
//             featimg {
//               childImageSharp {
//                 fluid(maxWidth: 300) {
//                   ...GatsbyImageSharpFluid
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// `
